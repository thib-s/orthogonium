import logging
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from flashlipschitz.layers import OrthoConv2d
from flashlipschitz.layers.conv.fast_block_ortho_conv import fast_matrix_conv
from flashlipschitz.layers.conv.fast_block_ortho_conv import transpose_kernel
from flashlipschitz.layers.conv.reparametrizers import OrthoParams


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class SDPBasedLipschitzResBlock(nn.Module):
    def __init__(self, cin, cout, inner_dim_factor, kernel_size=3, stride=2, **kwargs):
        super().__init__()
        inner_kernel_size = kernel_size - (stride - 1)
        self.skip_kernel_size = stride + (stride // 2)
        inner_dim = int(cout * inner_dim_factor)
        self.activation = nn.ReLU()
        self.stride = stride
        self.padding = kernel_size // 2
        self.kernel = nn.Parameter(
            torch.randn(inner_dim, cin, inner_kernel_size, inner_kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(1, inner_dim, 1, 1))
        self.q = nn.Parameter(torch.randn(inner_dim))

        nn.init.xavier_normal_(self.kernel)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.pre_conv = OrthoConv2d(
            cin, cin, kernel_size=stride, stride=1, bias=False, padding=0
        )
        self.post_conv = OrthoConv2d(
            cin, cout, kernel_size=stride, stride=stride, bias=False, padding=0
        )

    def compute_t(self):
        ktk = F.conv2d(self.kernel, self.kernel, padding=self.kernel.shape[-1] - 1)
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(t)
        return t

    def forward(self, x):
        # compute t
        t = self.compute_t()
        t = t.reshape(1, -1, 1, 1)
        # print(self.pre_conv.weight.shape, self.kernel.shape, self.post_conv.weight.shape)
        kernel_1a = fast_matrix_conv(self.pre_conv.weight, self.kernel, groups=1)
        kernel_1b = fast_matrix_conv(
            transpose_kernel(self.kernel, groups=1), self.post_conv.weight, groups=1
        )
        kernel_2 = fast_matrix_conv(
            self.pre_conv.weight, self.post_conv.weight, groups=1
        )
        # first branch
        # fuse pre conv with kernel
        res = F.conv2d(x, kernel_1a, padding=self.padding)
        res = res + self.bias
        res = t * self.activation(res)
        res = 2 * F.conv2d(res, kernel_1b, padding=self.padding, stride=self.stride)
        # residual branch
        x = F.conv2d(
            x, kernel_2, padding=self.skip_kernel_size // 2, stride=self.stride
        )
        # skip connection
        out = x - res
        return out


class SDPBasedLipschitzConv(nn.Module):
    def __init__(self, cin, inner_dim_factor, kernel_size=3, **kwargs):
        super().__init__()

        inner_dim = int(cin * inner_dim_factor)
        self.activation = nn.ReLU()

        self.padding = kernel_size // 2

        self.kernel = nn.Parameter(
            torch.randn(inner_dim, cin, kernel_size, kernel_size)
        )
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
        t = torch.abs(
            torch.einsum("i,ik,kj,j -> ij", q_inv, self.weight, self.weight.T, q)
        ).sum(1)
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


class SDPBasedLipschitzBCOPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inner_dim_factor: int,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "circular",
        ortho_params: OrthoParams = OrthoParams(),
    ):
        super().__init__()

        inner_dim = int(in_channels * inner_dim_factor)
        self.activation = nn.ReLU()

        if padding_mode not in ["circular", "zeros"]:
            raise ValueError("padding_mode must be either 'circular' or 'zeros'")
        if padding_mode == "circular":
            self.padding = 0  # will be handled by the padding function
        else:
            self.padding = kernel_size // 2

        self.in_conv = OrthoConv2d(
            in_channels,
            inner_dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            ortho_params=ortho_params,
        )
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, x):
        kernel = self.in_conv.weight
        # conv
        if self.padding_mode == "circular":
            x = F.pad(
                x,
                (
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                ),
                mode="circular",
            )
        res = F.conv2d(
            x, kernel, bias=self.in_conv.bias, padding=self.padding, groups=self.groups
        )
        # activation
        res = self.activation(res)
        # conv transpose
        if self.padding_mode == "circular":
            res = F.pad(
                res,
                (
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                ),
                mode="circular",
            )
        res = 2 * F.conv_transpose2d(
            res, kernel, padding=self.padding, groups=self.groups
        )
        # residual
        out = x - res
        return out
