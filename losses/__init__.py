import torch


class SquaredBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, t, eps=1e-8):
        log = lambda x: torch.log(torch.clamp(x, eps, 1.))
        return -(t*log(y**2) + (1-t)*log((1-y**2))).mean()
