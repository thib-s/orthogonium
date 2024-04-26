from typing import Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import parametrize as parametrize

from flashlipschitz.layers.conv.fast_block_ortho_conv import attach_bcop_weight, \
    fast_matrix_conv, conv_singular_values_numpy
from flashlipschitz.layers.conv.reparametrizers import BjorckParams
from flashlipschitz.layers.conv.rko_conv import attach_rko_weight


class BcopRkoConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = "same",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "circular",
        bjorck_params: BjorckParams = BjorckParams(),
    ):
        if dilation != 1:
            raise RuntimeError("dilation not supported")
        super(BcopRkoConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_channels = max(in_channels, out_channels)

        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise RuntimeError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (in_channels % groups != 0) and (out_channels % groups != 0):
            raise RuntimeError(
                "in_channels and out_channels must be divisible by groups"
            )
        if ((self.max_channels // groups) < 2) and (kernel_size != stride):
            raise RuntimeError("inner conv must have at least 2 channels")
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        self.intermediate_channels = in_channels
        ## oddly the following condition seems to not work
        # if in_channels >= out_channels * (stride**2):
        #     self.intermediate_channels = in_channels
        # else:
        #     self.intermediate_channels = out_channels * (stride**2)
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                in_channels // groups,
                kernel_size - (stride - 1),
                kernel_size - (stride - 1),
            ),
            groups,
            bjorck_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (out_channels, self.intermediate_channels // groups, stride, stride),
            groups,
            scale=1.0,
            bjorck_params=bjorck_params,
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self):
        if self.training:
            return fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        else:
            with parametrize.cached():
                return fast_matrix_conv(self.weight_1, self.weight_2, self.groups)

    def singular_values(self):
        if self.padding_mode != "circular":
            print(
                f"padding {self.padding} not supported, return min and max"
                f"singular values as if it was 'circular' padding "
                f"(overestimate the values)."
            )
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight_1.detach()
            .cpu()
            .reshape(
                self.groups,
                self.intermediate_channels // self.groups,
                self.in_channels // self.groups,
                self.kernel_size - (self.stride - 1),
                self.kernel_size - (self.stride - 1),
            )
            .numpy(),
            self._input_shape,
        )
        svs_2 = np.linalg.svd(
            self.weight_2.reshape(
                self.groups,
                self.out_channels // self.groups,
                self.intermediate_channels // self.groups * (self.stride**2),
            )
            .detach()
            .cpu()
            .numpy(),
            compute_uv=False,
        )
        sv_min = sv_min * svs_2.min()
        sv_max = sv_max * svs_2.max()
        stable_rank = 0.5 * stable_rank + 0.5 * (np.mean(svs_2) / (svs_2.max() ** 2))
        return sv_min, sv_max, stable_rank

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(BcopRkoConv2d, self).forward(X)


class BcopRkoConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        bjorck_params: BjorckParams = BjorckParams(),
    ):
        if dilation != 1:
            raise RuntimeError("dilation not supported")
        super(BcopRkoConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_channels = max(in_channels, out_channels)

        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise RuntimeError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (in_channels % groups != 0) and (out_channels % groups != 0):
            raise RuntimeError(
                "in_channels and out_channels must be divisible by groups"
            )
        if ((self.max_channels // groups) < 2) and (kernel_size != stride):
            raise RuntimeError("inner conv must have at least 2 channels")
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        self.intermediate_channels = out_channels
        # oddly the following condition seems to not work
        # if in_channels > out_channels :#* (stride**2):
        #     assert False
        #     self.intermediate_channels = in_channels * (stride**2)
        # else:
        #     self.intermediate_channels = out_channels
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                out_channels // groups,
                kernel_size - (stride - 1),
                kernel_size - (stride - 1),
            ),
            groups,
            bjorck_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (in_channels, self.intermediate_channels // groups, stride, stride),
            groups,
            scale=1.0,
            bjorck_params=bjorck_params,
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def singular_values(self):
        if self.padding_mode != "circular":
            print(
                f"padding {self.padding} not supported, return min and max"
                f"singular values as if it was 'circular' padding "
                f"(overestimate the values)."
            )
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight_1.detach()
            .cpu()
            .reshape(
                self.groups,
                self.intermediate_channels // self.groups,
                self.out_channels // self.groups,
                self.kernel_size - (self.stride - 1),
                self.kernel_size - (self.stride - 1),
            )
            .numpy(),
            self._input_shape,
        )
        svs_2 = np.linalg.svd(
            self.weight_2.reshape(
                self.groups,
                self.in_channels // self.groups,
                self.intermediate_channels // self.groups * (self.stride**2),
            )
            .detach()
            .cpu()
            .numpy(),
            compute_uv=False,
        )
        sv_min = sv_min * svs_2.min()
        sv_max = sv_max * svs_2.max()
        stable_rank = 0.5 * stable_rank + 0.5 * (np.mean(svs_2) / (svs_2.max() ** 2))
        return sv_min, sv_max, stable_rank

    @property
    def weight(self):
        if self.training:
            kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        else:
            with parametrize.cached():
                kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        # kernel = kernel.view(
        #     self.groups,
        #     self.out_channels // self.groups,
        #     self.in_channels // self.groups,
        #     self.kernel_size,
        #     self.kernel_size,
        # )
        # kernel = kernel.transpose(-3,-4).flip([-2, -1])
        # kernel = kernel.view(
        #     self.in_channels,
        #     self.out_channels // self.groups,
        #     self.kernel_size,
        #     self.kernel_size,
        #
        # )
        return kernel

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(BcopRkoConvTranspose2d, self).forward(X)
