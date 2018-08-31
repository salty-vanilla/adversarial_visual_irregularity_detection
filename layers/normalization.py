import torch


class PixelNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, x):
        norm = torch.mean(x**2,
                          dim=1,
                          keepdim=True)
        return x / (norm + self.eps) ** 0.5


class LayerNorm(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = 1e-8

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
