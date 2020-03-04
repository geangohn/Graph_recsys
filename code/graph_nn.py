import torch
import torch.nn as nn


class GCNConv(nn.Module):
    """Graph Convolutional Layer

    Input
    -----
    A - adjacency matrix
    in_channels - number of input features
    out_channels - number of output features

    Output
    ------
    output = RELU(A_hat * X * W)
    """
    
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A + torch.eye(A.size(0))
        self.D = torch.diag(torch.sum(A, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))

    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))

        return out