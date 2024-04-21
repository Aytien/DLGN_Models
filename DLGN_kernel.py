import torch 
from torch import nn
import numpy as np
from copy import deepcopy
from DLGN_enums import LossTypes

def ScaledSig(x, beta):
    y = beta*x
    S = nn.Sigmoid()
    return S(y)


class DLGN_Kernel(nn.Module):
    def __init__(self, num_data, dim_in,  width, depth, beta = 4, alpha_init='zero', loss_fn_type=LossTypes.HINGE):
        super().__init__()
        self.num_data = num_data
        self.beta = beta
        self.dim_in = dim_in
        self.width = width
        self.depth = depth
        self.ainit = alpha_init
        self.loss_fn_type = loss_fn_type
        sigma = 1/np.sqrt(width)
        self.gating_layers = nn.ParameterList([nn.Parameter(sigma*torch.randn(dim_in if i == 0 else width, width)) for i in range(depth)])
        if self.ainit == 'zero':
            self.alphas = nn.Parameter(torch.zeros(num_data), requires_grad=False)
        else:
            self.alphas = nn.Parameter(torch.randn(num_data)/np.sqrt(num_data), requires_grad=False)

    # def get_weights(self):
    #     A = [self.gating_layers[0]]
    #     for i in range(1,self.depth):
    #         A.append(A[-1]@self.gating_layers[i])
    #     return torch.vstack(A)


    def forward(self, inp, data):
        output_kernel = self.npk_forward(inp, data)
        return output_kernel @ self.alphas
    
    def npk_forward(self, inp, data):  
        
        data_gate_matrix = data @ self.gating_layers[0]
        data_gate_score = ScaledSig(data_gate_matrix, self.beta)
        inp_gate_matrix = inp @ self.gating_layers[0]
        inp_gate_score = ScaledSig(inp_gate_matrix, self.beta)
        output_kernel =  (inp_gate_score @ data_gate_score.T)

        for i in range(1,self.depth):
            data_gate_matrix = data_gate_matrix @ self.gating_layers[i]
            inp_gate_matrix = inp_gate_matrix @ self.gating_layers[i]
            data_gate_score = ScaledSig(data_gate_matrix, self.beta)
            inp_gate_score = ScaledSig(inp_gate_matrix, self.beta)
            output_kernel *= (inp_gate_score @ data_gate_score.T)/self.width

        return output_kernel

    def get_npk(self, X, Y):

        device = self.gating_layers[0].device
        if type(X) is np.ndarray:
            X = torch.tensor(X, dtype=torch.float32).to(device)
            Y = torch.tensor(Y, dtype=torch.float32).to(device)
        else:
            X = X.to(device)
            Y = Y.to(device)

        output_kernel = self.npk_forward(X, Y)

        return output_kernel.detach().cpu().numpy()


    
    def log_features_dlgn(self,bias=False):   ## Don't use it for DLGN self
        w_list = []
        b_list = []
        for name, param in self.named_parameters():
            for i in range(0,self.depth):
                param_data = deepcopy(param.data)
                if name == 'gating_layers.'+str(i)+'.weight':
                    w_list.append(param_data)
                if bias:
                    if name == 'gating_layers.'+str(i)+'.bias':
                        b_list.append(param_data)

        Feature_list = [w_list[0]]

        for w in w_list[1:]:
            Feature_list.append(w @ Feature_list[-1])

        features = torch.cat(Feature_list, axis = 0).to("cpu")

        return features