import warnings
from typing import Union

import numpy as np
from torch import nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import parametrize as parametrize

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import (
    attach_bcop_weight,
)
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_matrix_conv
from orthogonium.layers.conv.AOC.rko_conv import attach_rko_weight
from orthogonium.reparametrizers import OrthoParams


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
        if (self.dilation[0] != 1 or self.dilation[1] != 1) and (
            self.stride[0] != 1 or self.stride[1] != 1
        ):
            raise RuntimeError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        # raise runtime error if kernel size >= stride
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise RuntimeError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (in_channels % groups != 0) and (out_channels % groups != 0):
            raise RuntimeError(
                "in_channels and out_channels must be divisible by groups"
            )
        if (
            ((max(in_channels, out_channels) // groups) < 2)
            and (self.kernel_size[0] != self.stride[0])
            and (self.kernel_size[1] != self.stride[1])
        ):
            raise RuntimeError("inner conv must have at least 2 channels")
        self.intermediate_channels = max(
            in_channels, out_channels // (self.stride[0] * self.stride[1])
        )
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                in_channels // groups,
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            ),
            groups,
            ortho_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (
                out_channels,
                self.intermediate_channels // groups,
                self.stride[0],
                self.stride[1],
            ),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )

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

    def singular_values(self):
        """
        The estimation of the singular values under striding is not trivial.
        This approximation is obtained by composing the singular values of the
        two convolutions. The singular values of the first convolution are computed
        using the FFT method, while the singular values of the second convolution
        are computed using the SVD method. The product of the singular values of
        can lead to an overestimation of the largest singular values, and an underestimation
        of the smallest singular values. However when both convolutions are orthogonal,
        the estimation is exact. In other words this method is a good sanity check to
        ensure that the convolutions are orthogonal.
        """
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
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            )
            .numpy(),
            self._input_shape,
        )
        svs_2 = np.linalg.svd(
            self.weight_2.reshape(
                self.groups,
                self.out_channels // self.groups,
                (self.intermediate_channels // self.groups)
                * (self.stride[0] * self.stride[1]),
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
        ortho_params: OrthoParams = OrthoParams(),
    ):
        """As BcopRkoConv2d handle native striding with explicit kernel. It unlocks
        the possibility to use the same parametrization for transposed convolutions.
        This class uses the same interface as the ConvTranspose2d class.

        Unfortunately, circular padding is not supported for the transposed convolution.
        But unit testing have shown that the convolution is still orthogonal when
         `out_channels * (self.stride[0]*self.stride[1]) > in_channels`.
        """
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

        # raise runtime error if kernel size >= stride
        if (self.dilation[0] != 1 or self.dilation[1] != 1) and (
            self.stride[0] != 1 or self.stride[1] != 1
        ):
            raise RuntimeError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise RuntimeError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (in_channels % groups != 0) and (out_channels % groups != 0):
            raise RuntimeError(
                "in_channels and out_channels must be divisible by groups"
            )
        if (
            ((max(in_channels, out_channels) // groups) < 2)
            and (self.kernel_size[0] != self.stride[0])
            and (self.kernel_size[1] != self.stride[1])
        ):
            raise RuntimeError("inner conv must have at least 2 channels")
        if out_channels * (self.stride[0] * self.stride[1]) >= in_channels:
            self.intermediate_channels = max(
                in_channels // (self.stride[0] * self.stride[1]), out_channels
            )
        else:
            self.intermediate_channels = out_channels
            # raise warning because this configuration don't yield orthogonal
            # convolutions
            warnings.warn(
                "This configuration does not yield orthogonal convolutions due to "
                "padding issues: pytorch does not implement circular padding for "
                "transposed convolutions",
                RuntimeWarning,
            )
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                out_channels // groups,
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            ),
            groups,
            ortho_params=ortho_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (
                in_channels,
                self.intermediate_channels // groups,
                self.stride[0],
                self.stride[1],
            ),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )

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
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            )
            .numpy(),
            self._input_shape,
        )
        svs_2 = np.linalg.svd(
            self.weight_2.reshape(
                self.groups,
                self.in_channels // self.groups,
                (self.intermediate_channels // self.groups)
                * (self.stride[0] * self.stride[1]),
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
        return kernel.contiguous()

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(BcopRkoConvTranspose2d, self).forward(X)
