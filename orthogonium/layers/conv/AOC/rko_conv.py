import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.reparametrizers import OrthoParams


class RKOParametrizer(nn.Module):
    def __init__(
        self, kernel_shape, groups, scale, ortho_params: OrthoParams = OrthoParams()
    ):
        super(RKOParametrizer, self).__init__()
        self.kernel_shape = kernel_shape
        self.groups = groups
        out_channels, in_channels, k1, k2 = kernel_shape
        in_channels *= groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k1 = k1
        self.k2 = k2
        self.scale = scale
        self.register_module(
            "pi",
            ortho_params.spectral_normalizer(
                weight_shape=(
                    self.groups,
                    out_channels // self.groups,
                    in_channels // groups * k1 * k2,
                ),
            ),
        )
        self.register_module(
            "bjorck",
            ortho_params.orthogonalizer(
                weight_shape=(
                    self.groups,
                    out_channels // self.groups,
                    in_channels // groups * k1 * k2,
                ),
            ),
        )

    def forward(self, X):
        X = X.view(
            self.groups,
            self.out_channels // self.groups,
            (self.in_channels // self.groups) * self.k1 * self.k2,
        )
        X = self.pi(X)
        X = self.bjorck(X)
        X = X.reshape(
            self.out_channels, self.in_channels // self.groups, self.k1, self.k2
        )
        return X * self.scale

    def right_inverse(self, X):
        return X


def attach_rko_weight(
    layer,
    weight_name,
    kernel_shape,
    groups,
    scale=None,
    ortho_params: OrthoParams = OrthoParams(),
):
    out_channels, in_channels, kernel_size, k2 = kernel_shape
    in_channels *= groups
    if scale is None:
        scale = 1 / math.sqrt(kernel_size * kernel_size)

    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(
            torch.Tensor(torch.randn(*kernel_shape)),
            requires_grad=True,
        ),
    )
    weight = getattr(layer, weight_name)
    torch.nn.init.normal_(weight)
    parametrize.register_parametrization(
        layer,
        weight_name,
        RKOParametrizer(
            kernel_shape,
            groups,
            scale,
            ortho_params=ortho_params,
        ),
    )


class RKOConv2d(nn.Conv2d):
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
        super(RKOConv2d, self).__init__(
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
        if (
            True  # (self.out_channels >= self.in_channels) # investigate why it don't work
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthogonal convolutions is empty in this setting."
            )
        torch.nn.init.orthogonal_(self.weight)
        self.scale = 1 / math.sqrt(
            math.ceil(self.kernel_size[0] / self.stride[0])
            * math.ceil(self.kernel_size[1] / self.stride[1])
        )
        parametrize.register_parametrization(
            self,
            "weight",
            RKOParametrizer(
                kernel_shape=(
                    out_channels,
                    in_channels // self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
                groups=self.groups,
                scale=self.scale,
                ortho_params=ortho_params,
            ),
        )

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(RKOConv2d, self).forward(X)

    def singular_values(self):
        if (self.stride[0] == self.kernel_size[0]) and (
            self.stride[1] == self.kernel_size[1]
        ):
            svs = np.linalg.svd(
                self.weight.reshape(
                    self.groups,
                    self.out_channels // self.groups,
                    (self.in_channels // self.groups)
                    * (self.stride[0] * self.stride[1]),
                )
                .detach()
                .cpu()
                .numpy(),
                compute_uv=False,
            )
            sv_min = svs.min()
            sv_max = svs.max()
            stable_rank = (np.mean(svs) ** 2) / (svs.max() ** 2)
            return sv_min, sv_max, stable_rank
        elif self.stride[0] > 1 or self.stride[1] > 1:
            raise RuntimeError(
                "Not able to compute singular values for this configuration"
            )
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach()
            .cpu()
            .view(
                self.groups,
                self.out_channels // self.groups,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
            .numpy(),
            self._input_shape,
        )
        return sv_min, sv_max, stable_rank


class RkoConvTranspose2d(nn.ConvTranspose2d):
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
        super(RkoConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding if padding_mode == "zeros" else 0,
            output_padding,
            groups,
            bias,
            dilation,
            "zeros",
        )
        self.real_padding_mode = padding_mode
        if padding == "same":
            padding = self._calculate_same_padding()
        self.real_padding = self._standardize_padding(padding)
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthogonal convolutions is empty in this setting."
            )
        if (
            (self.out_channels <= self.in_channels)
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        if (
            self.stride[0] != self.kernel_size[0]
            or self.stride[1] != self.kernel_size[1]
        ):
            self.scale = 1 / math.sqrt(
                math.ceil(self.kernel_size[0] / self.stride[0])
                * math.ceil(self.kernel_size[1] / self.stride[1])
            )
        else:
            self.scale = 1
        del self.weight
        attach_rko_weight(
            self,
            "weight",
            (in_channels, out_channels // groups, self.stride[0], self.stride[1]),
            groups,
            scale=self.scale,
            ortho_params=ortho_params,
        )

    def _calculate_same_padding(self) -> tuple:
        """Calculate padding for 'same' mode."""
        return (
            int(
                np.ceil(
                    (self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0])
                    / 2
                )
            ),
            int(
                np.floor(
                    (self.dilation[0] * (self.kernel_size[0] - 1) + 1 - self.stride[0])
                    / 2
                )
            ),
            int(
                np.ceil(
                    (self.dilation[1] * (self.kernel_size[1] - 1) + 1 - self.stride[1])
                    / 2
                )
            ),
            int(
                np.floor(
                    (self.dilation[1] * (self.kernel_size[1] - 1) + 1 - self.stride[1])
                    / 2
                )
            ),
        )

    def _standardize_padding(self, padding: _size_2_t) -> tuple:
        """Ensure padding is always a tuple."""
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(padding, tuple):
            if len(padding) == 2:
                padding = (padding[0], padding[0], padding[1], padding[1])
            return padding
        raise ValueError(f"padding must be int or tuple, got {type(padding)} instead")

    def singular_values(self):
        if (self.stride[0] == self.kernel_size[0]) and (
            self.stride[1] == self.kernel_size[1]
        ):
            svs = np.linalg.svd(
                self.weight.reshape(
                    self.groups,
                    self.in_channels // self.groups,
                    (self.out_channels // self.groups)
                    * (self.stride[0] * self.stride[1]),
                )
                .detach()
                .cpu()
                .numpy(),
                compute_uv=False,
            )
            sv_min = svs.min()
            sv_max = svs.max()
            stable_rank = (np.mean(svs) ** 2) / (svs.max() ** 2)
            return sv_min, sv_max, stable_rank
        elif self.stride[0] > 1 or self.stride[1] > 1:
            raise RuntimeError(
                "Not able to compute singular values for this configuration"
            )
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach()
            .cpu()
            .view(
                self.groups,
                self.out_channels // self.groups,
                self.in_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
            .numpy(),
            self._input_shape,
        )
        return sv_min, sv_max, stable_rank

    def forward(self, X):
        self._input_shape = X.shape[2:]
        if self.real_padding_mode != "zeros":
            X = nn.functional.pad(X, self.real_padding, self.real_padding_mode)
            y = nn.functional.conv_transpose2d(
                X,
                self.weight,
                self.bias,
                self.stride,
                (
                    (
                        -self.stride[0]
                        + self.dilation[0] * (self.kernel_size[0] - 1)
                        + 1
                    ),
                    (
                        -self.stride[1]
                        + self.dilation[1] * (self.kernel_size[1] - 1)
                        + 1
                    ),
                ),
                self.output_padding,
                self.groups,
                dilation=self.dilation,
            )
            return y
        else:
            return super(RkoConvTranspose2d, self).forward(X)
