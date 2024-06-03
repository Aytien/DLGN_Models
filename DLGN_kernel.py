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
    def __init__(self, num_data, dim_in,  width, depth, beta = 4, alpha_init='zero', BN=False, feat = 'cf'):
        super().__init__()
        self.num_data = num_data
        self.beta = beta
        self.dim_in = dim_in
        self.width = width
        self.depth = depth
        self.ainit = alpha_init
        self.BN = BN
        sigma = 1/np.sqrt(width)
        self.feat = feat
        if self.feat == 'sf':
            self.gating_layers = nn.ParameterList([nn.Parameter(sigma*torch.randn(dim_in, width)) for i in range(depth)])
        elif self.feat == 'cf':
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


    # def forward(self, inp, data):
    #     output_kernel = self.npk_forward(inp, data)
    #     preds = output_kernel @ self.alphas
    #     del output_kernel
    #     with torch.cuda.device(self.gating_layers[0].device):
    #         torch.cuda.empty_cache()
    #     return preds
    
    def forward(self, inp, data):
        data_gate_matrix = data @ self.gating_layers[0]
        inp_gate_matrix = inp @ self.gating_layers[0]

        if self.BN:
            data_gate_matrix /= torch.norm(self.gating_layers[0], dim=0, keepdim=True)
            inp_gate_matrix /= torch.norm(self.gating_layers[0], dim=0, keepdim=True)
        data_gate_score = ScaledSig(data_gate_matrix, self.beta)
        inp_gate_score = ScaledSig(inp_gate_matrix, self.beta)
        output_kernel =  (inp_gate_score @ data_gate_score.T)

        for i in range(1,self.depth):
            if self.feat == 'sf':
                data_gate_matrix = data @ self.gating_layers[i]
                inp_gate_matrix = inp @ self.gating_layers[i]
            elif self.feat == 'cf':
                data_gate_matrix = data_gate_matrix @ self.gating_layers[i]
                inp_gate_matrix = inp_gate_matrix @ self.gating_layers[i]
            if self.BN:
                data_gate_matrix /= torch.norm(self.gating_layers[i], dim=0, keepdim=True)
                inp_gate_matrix /= torch.norm(self.gating_layers[i], dim=0, keepdim=True)
            data_gate_score = ScaledSig(data_gate_matrix, self.beta)
            inp_gate_score = ScaledSig(inp_gate_matrix, self.beta)
            output_kernel *= (inp_gate_score @ data_gate_score.T)/self.width
        preds = output_kernel @ self.alphas
        del output_kernel
        with torch.cuda.device(self.gating_layers[0].device):
            torch.cuda.empty_cache()
        return preds

    def npk_forward(self, inp, data):  
        
        data_gate_matrix = data @ self.gating_layers[0]
        inp_gate_matrix = inp @ self.gating_layers[0]

        if self.BN:
            data_gate_matrix /= torch.norm(self.gating_layers[0], dim=0, keepdim=True)
            inp_gate_matrix /= torch.norm(self.gating_layers[0], dim=0, keepdim=True)
        data_gate_score = ScaledSig(data_gate_matrix, self.beta)
        inp_gate_score = ScaledSig(inp_gate_matrix, self.beta)
        output_kernel =  (inp_gate_score @ data_gate_score.T)

        for i in range(1,self.depth):
            if self.feat == 'sf':
                data_gate_matrix = data @ self.gating_layers[i]
                inp_gate_matrix = inp @ self.gating_layers[i]
            elif self.feat == 'cf':
                data_gate_matrix = data_gate_matrix @ self.gating_layers[i]
                inp_gate_matrix = inp_gate_matrix @ self.gating_layers[i]
            if self.BN:
                data_gate_matrix /= torch.norm(self.gating_layers[i], dim=0, keepdim=True)
                inp_gate_matrix /= torch.norm(self.gating_layers[i], dim=0, keepdim=True)
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
        output_kernel = output_kernel.cpu().detach().numpy()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        return output_kernel 


    
    def log_features(self,bias=False):   ## Don't use it for DLGN self
        w_list = []
        b_list = []
        for name, param in self.named_parameters():
            for i in range(0,self.depth):
                if name == 'gating_layers.'+str(i):
                    w_list.append(param.data)
                if bias:
                    if name == 'gating_layers.'+str(i)+'.bias':
                        b_list.append(param.data)

        Feature_list = [w_list[0].T/torch.linalg.norm(w_list[0].T, ord=2, dim=1).reshape(-1,1)]

        for w in w_list[1:]:
            if self.feat == 'cf':
                Feature_list.append(w.T @ Feature_list[-1])
            elif self.feat == 'sf':
                Feature_list.append(w.T)
            Feature_list[-1] = Feature_list[-1]/torch.linalg.norm(Feature_list[-1], ord=2, dim=1).reshape(-1,1)


        features = torch.cat(Feature_list, axis = 0).to("cpu")

        return features