import torch
import torch.nn as nn


class LayerCentering2D(nn.Module):
    def __init__(self, num_features):
        super(LayerCentering2D, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros((1, num_features, 1, 1)), requires_grad=True
        )

    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return x - mean + self.bias


class BatchCentering2D(nn.Module):
    def __init__(self, num_features, momentum=0.10):
        super(BatchCentering2D, self).__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(1))
        self.bias = nn.Parameter(
            torch.zeros((1, num_features, 1, 1)), requires_grad=True
        )

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, -2, -1), keepdim=True)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
            return x - self.running_mean.detach() + self.bias
        else:
            return x - self.running_mean + self.bias
