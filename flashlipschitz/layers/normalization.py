import torch
import torch.nn as nn


class LayerCentering(nn.Module):
    def __init__(self, dim=-1):
        super(LayerCentering, self).__init__()
        self.dim = dim

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        return x - mean


class BatchCentering(nn.Module):
    def __init__(self, dim=-1, momentum=0.1):
        super(BatchCentering, self).__init__()
        self.dim = dim
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(1))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=self.dim, keepdim=True)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
        return x - self.running_mean
