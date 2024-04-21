import torch
import torch.nn as nn
import numpy as np
# import torch.nn.functional as F

class DLGN_VT(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, num_hidden_nodes=[], beta=30, mode='pwc',value_scale=500.,BN=False,prod='op',feat='sf'):		
        super(DLGN_VT, self).__init__()
        self.num_hidden_layers = len(num_hidden_nodes)
        self.beta=beta  # Soft gating parameter
        self.mode = mode
        self.BN = BN
        self.prod = prod
        self.feat = feat
        self.num_nodes=[input_dim]+num_hidden_nodes+[output_dim]
        self.gating_layers=nn.ModuleList()
        if self.prod=='op':
            self.value_layers=nn.Parameter(torch.randn([1]+num_hidden_nodes)/value_scale)
        elif self.prod=='ip':
            self.value_layers=nn.Parameter(torch.randn(num_hidden_nodes[0])/value_scale)
        self.num_layer = len(num_hidden_nodes)
        self.num_hidden_nodes = num_hidden_nodes
        for i in range(self.num_hidden_layers+1):
            if i!=self.num_hidden_layers:
                if self.feat == 'sf':
                    temp = nn.Linear(self.num_nodes[0], self.num_nodes[i+1], bias=False)
                elif self.feat == 'cf':
                    temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1], bias=False)
                self.gating_layers.append(temp)

    def set_parameters_with_mask(self, to_copy, parameter_masks):
		# self and to_copy are DLGN_FC objects with same architecture
		# parameter_masks is compatible with dict(to_copy.named_parameters())
        for (name, copy_param) in to_copy.named_parameters():
            copy_param = copy_param.clone().detach()
            orig_param  = self.state_dict()[name]
            if name in parameter_masks:
                param_mask = parameter_masks[name]>0
                orig_param[param_mask] = copy_param[param_mask]
            else:
                orig_param = copy_param.data.detach()
	

    def return_gating_functions(self):
        effective_weights = []
        for i in range(self.num_hidden_layers):
            # if self.feat=='cf' and i!=0:
            # 	curr_weight = self.gating_layers[i].weight.detach().clone()@curr_weight
            curr_weight = self.gating_layers[i].weight.detach().clone()
            if self.BN:
                curr_weight /= torch.norm(curr_weight, dim=1, keepdim=True)
            effective_weights.append(curr_weight)
        return effective_weights
        # effective_weights (and effective biases) is a list of size num_hidden_layers
                            

    def forward(self, x): 
        # Values is a list of size 1+num_hidden_layers+1
        # Gate_scores is a list of size 1+num_hidden_layers
        cp = self.npk_forward(x)
        if self.prod=='op':
            return torch.sum(cp*self.value_layers, dim=tuple(range(1,self.num_layer+1)))
        elif self.prod=='ip':
            return torch.sum(cp*self.value_layers, dim=1)

    def get_npk(self, X, Y):
        # X, Y are np arrays of size n x d, n x d
        # Returns the NPK matrix of size n x n
        device = self.gating_layers[0].weight.device
        if type(X) is np.ndarray:
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y = torch.tensor(Y, dtype=torch.float32).to(device)
        else:
            X = X.to(device)
            Y = Y.to(device)

        gate_scores_x = self.get_gate_scores(X)
        gate_scores_y = self.get_gate_scores(Y)
        if self.prod=='op':
            kval = torch.ones(X.shape[0], Y.shape[0]).to(device)
            for i in range(len(gate_scores_x)):
                kval = kval*torch.matmul(gate_scores_x[i].to(device), gate_scores_y[i].to(device).T)
            return kval.detach().cpu().numpy()
        
    
    def get_gate_scores(self, x):
        gate_scores = []
        h = x
        for i in range(len(self.gating_layers)):
            if self.BN:
                if self.feat=='cf':
                    h = self.gating_layers[i](h)/torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T
                    gate_score = torch.sigmoid( self.beta*h)
                else:
                    gate_score = torch.sigmoid( self.beta*self.gating_layers[i](x)/torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T)
            else:
                if self.feat=='cf':
                    h = self.gating_layers[i](h)
                    gate_score = torch.sigmoid( self.beta*h)
                else:
                    gate_score = torch.sigmoid( self.beta*self.gating_layers[i](x))
            
            gate_scores.append(gate_score)
            if self.feat=='nf':
                h=gate_score
        return gate_scores
	

    def npk_forward(self, x):
        h=x

        for i in range(self.num_hidden_layers):
            if self.prod=='op':
                fiber = [len(h)]+[1]*self.num_hidden_layers
                fiber[i+1] = self.num_hidden_nodes[i]
                fiber = tuple(fiber)
            
            if self.BN:
                if self.feat=='cf':
                    h = self.gating_layers[i](h)/torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T
                    gate_score = torch.sigmoid( self.beta*h)
                else:
                    gate_score = torch.sigmoid( self.beta*self.gating_layers[i](h)/torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T)  
                
            else:
                if self.feat=='cf':
                    h = self.gating_layers[i](h)
                    gate_score = torch.sigmoid( self.beta*h)
                else:
                    gate_score = torch.sigmoid( self.beta*self.gating_layers[i](h)) # batch * m

            if self.feat=='nf':
                h=gate_score
                
            if self.prod=='op':
                gate_score = gate_score.reshape(fiber) # batch * 1 * 1 * 1 *1 with one of the ones replaced by an m

            if i==0:
                cp = gate_score
            else:
                cp = cp*gate_score #batch * m^{i} -> batch * m^{i+1} 

        return cp