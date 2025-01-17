import warnings
from typing import Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import parametrize as parametrize
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_matrix_conv
from orthogonium.layers.conv.AOC.rko_conv import attach_rko_weight
from orthogonium.layers.conv.adaptiveSOC.fast_skew_ortho_conv import (
    attach_soc_weight,
    ExpParams,
)
from orthogonium.reparametrizers import OrthoParams


class SOCRkoConv2d(nn.Conv2d):
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
        exp_params: ExpParams = ExpParams(),
        ortho_params: OrthoParams = OrthoParams(),
    ):
        """
        This class handle native striding by combining a k-s x k-s convolution
        (without stride) with a sxs convolution (with stride s). This results in
        a kxk convolution with stride s. The first convolution is orthogonalized
        using BCOP and the second one is orthogonalized using RKO. By setting appropriate
        number of channels in the intermediate layer, the resulting convolution is
        orthogonal.

        It is advised to use the OrthogonalConv2d class instead of this one, as it
        instanciates the most efficient convolution for the given configuration.
        """
        super(SOCRkoConv2d, self).__init__(
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
        self.max_channels = max(in_channels, out_channels)
        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if ((self.max_channels // groups) < 2) and (kernel_size != stride):
            raise ValueError("inner conv must have at least 2 channels")
        self.intermediate_channels = max(in_channels, out_channels // stride**2)
        del self.weight
        int_kernel_size = kernel_size - (stride - 1)
        if int_kernel_size % 2 == 0:
            if int_kernel_size <= 2:
                int_kernel_size += 1
            else:
                int_kernel_size -= 1
            # warn user that kernel size changed
            warnings.warn(
                f"kernel size changed from {kernel_size} to {int_kernel_size} "
                f"as even kernel size is not supported for SOC.",
                RuntimeWarning,
            )
        attach_soc_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                in_channels // groups,
                int_kernel_size,
                int_kernel_size,
            ),
            groups,
            exp_params=exp_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (out_channels, self.intermediate_channels // groups, stride, stride),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )
        self.kernel_size = self.weight.shape[-2:]

    @property
    def weight(self):
        if self.training:
            return fast_matrix_conv(
                self.weight_1, self.weight_2, self.groups
            ).contiguous()
        else:
            with parametrize.cached():
                return fast_matrix_conv(
                    self.weight_1, self.weight_2, self.groups
                ).contiguous()


class SOCRkoConvTranspose2d(nn.ConvTranspose2d):
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
        exp_params: ExpParams = ExpParams(),
        ortho_params: OrthoParams = OrthoParams(),
    ):
        """As BcopRkoConv2d handle native striding with explicit kernel. It unlocks
        the possibility to use the same parametrization for transposed convolutions.
        This class uses the same interface as the ConvTranspose2d class.

        Unfortunately, circular padding is not supported for the transposed convolution.
        But unit testing have shown that the convolution is still orthogonal when
         `out_channels * (stride**2) > in_channels`.
        """
        super(SOCRkoConvTranspose2d, self).__init__(
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
        self.max_channels = max(in_channels, out_channels)

        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if ((self.max_channels // groups) < 2) and (kernel_size != stride):
            raise ValueError("inner conv must have at least 2 channels")
        if out_channels * (stride**2) >= in_channels:
            self.intermediate_channels = max(in_channels // (stride**2), out_channels)
        else:
            self.intermediate_channels = out_channels
        del self.weight
        int_kernel_size = kernel_size - (stride - 1)
        if int_kernel_size % 2 == 0:
            if int_kernel_size <= 2:
                int_kernel_size += 1
            else:
                int_kernel_size -= 1
            # warn user that kernel size changed
            warnings.warn(
                f"kernel size changed from {kernel_size} to {int_kernel_size} "
                f"as even kernel size is not supported for SOC.",
                RuntimeWarning,
            )
        attach_soc_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                out_channels // groups,
                int_kernel_size,
                int_kernel_size,
            ),
            groups,
            exp_params=exp_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (in_channels, self.intermediate_channels // groups, stride, stride),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )
        self.kernel_size = self.weight.shape[-2:]

    @property
    def weight(self):
        if self.training:
            kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        else:
            with parametrize.cached():
                kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        return kernel.contiguous()
