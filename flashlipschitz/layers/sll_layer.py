import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class SDPBasedLipschitzConv(nn.Module):

    def __init__(self, cin, inner_dim, kernel_size=3, **kwargs):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else cin
        self.activation = nn.ReLU()

        self.padding = kernel_size // 2

        self.kernel = nn.Parameter(torch.randn(inner_dim, cin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        t = t.reshape(1, -1, 1, 1)
        res = F.conv2d(x, self.kernel, padding=1)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.conv_transpose2d(res, self.kernel, padding=1)
        out = x - res
        return out


class SDPBasedLipschitzDense(nn.Module):

    def __init__(self, in_features, out_features, inner_dim, **kwargs):
        super().__init__()

        inner_dim = inner_dim if inner_dim != -1 else in_features
        self.activation = nn.ReLU()

        self.weight = nn.Parameter(torch.empty(inner_dim, in_features))
        self.bias = nn.Parameter(torch.empty(1, inner_dim))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.weight)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def compute_t(self):
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = torch.abs(torch.einsum('i,ik,kj,j -> ij', q_inv, self.weight, self.weight.T, q)).sum(1)
        t = safe_inv(t)
        return t

    def forward(self, x):
        t = self.compute_t()
        res = F.linear(x, self.weight)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.linear(res, self.weight.T)
        out = x - res
        return out
