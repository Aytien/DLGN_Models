import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        '''
        y_pred: (batch_size, 1), Real numbers
        y_true: (batch_size, 1), {1, 0}
        '''
        return torch.mean(torch.clamp(1 - y_pred * (2*y_true-1), min=0))