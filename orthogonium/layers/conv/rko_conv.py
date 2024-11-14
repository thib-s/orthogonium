import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from orthogonium.layers.conv.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.reparametrizers import (
    BatchedBjorckOrthogonalization,
    # OrthoParams,
)
from orthogonium.layers.conv.reparametrizers import (
    BatchedPowerIteration,
)
from orthogonium.layers.conv.reparametrizers import L2Normalize
from orthogonium.layers.conv.reparametrizers import OrthoParams


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
        X = X.view(self.out_channels, self.in_channels // self.groups, self.k1, self.k2)
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
        # torch.nn.init.orthogonal_(self.weight)
        if stride != kernel_size:
            self.scale = 1 / math.sqrt(kernel_size * kernel_size)
        else:
            self.scale = 1
        parametrize.register_parametrization(
            self,
            "weight",
            RKOParametrizer(
                kernel_shape=(
                    out_channels,
                    in_channels // self.groups,
                    kernel_size,
                    kernel_size,
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
        if isinstance(self.kernel_size, tuple):
            kernel_size = self.kernel_size
        else:
            kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, tuple):
            stride = self.stride
        else:
            stride = (self.stride, self.stride)
        if (stride[0] == kernel_size[0]) and (stride[1] == kernel_size[1]):
            svs = np.linalg.svd(
                self.weight.reshape(
                    self.groups,
                    self.out_channels // self.groups,
                    (self.in_channels // self.groups) * (stride[0] * stride[1]),
                )
                .detach()
                .cpu()
                .numpy(),
                compute_uv=False,
            )
            sv_min = svs.min()
            sv_max = svs.max()
            stable_rank = np.mean(svs) / (svs.max() ** 2)
            return sv_min, sv_max, stable_rank
        elif stride[0] > 1 or stride[1] > 1:
            raise RuntimeError(
                "Not able to compute singular values for this " "configuration"
            )
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach()
            .cpu()
            .view(
                self.groups,
                self.out_channels // self.groups,
                self.in_channels // self.groups,
                kernel_size[0],
                kernel_size[1],
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
        if dilation != 1:
            raise RuntimeError("dilation not supported")
        super(RkoConvTranspose2d, self).__init__(
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
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        del self.weight
        attach_rko_weight(
            self,
            "weight",
            (in_channels, out_channels // groups, stride, stride),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def singular_values(self):
        if isinstance(self.kernel_size, tuple):
            kernel_size = self.kernel_size
        else:
            kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.stride, tuple):
            stride = self.stride
        else:
            stride = (self.stride, self.stride)
        if (stride[0] == kernel_size[0]) and (stride[1] == kernel_size[1]):
            svs = np.linalg.svd(
                self.weight.reshape(
                    self.groups,
                    self.in_channels // self.groups,
                    (self.out_channels // self.groups) * (stride[0] * stride[1]),
                )
                .detach()
                .cpu()
                .numpy(),
                compute_uv=False,
            )
            sv_min = svs.min()
            sv_max = svs.max()
            stable_rank = np.mean(svs) / (svs.max() ** 2)
            return sv_min, sv_max, stable_rank
        elif stride[0] > 1 or stride[1] > 1:
            raise RuntimeError(
                "Not able to compute singular values for this " "configuration"
            )
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach()
            .cpu()
            .view(
                self.groups,
                self.out_channels // self.groups,
                self.in_channels // self.groups,
                kernel_size[0],
                kernel_size[1],
            )
            .numpy(),
            self._input_shape,
        )
        return sv_min, sv_max, stable_rank

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(RkoConvTranspose2d, self).forward(X)


class OrthoLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ortho_params: OrthoParams = OrthoParams(),
    ):
        super(OrthoLinear, self).__init__(in_features, out_features, bias=bias)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            ortho_params.spectral_normalizer(
                weight_shape=(self.out_features, self.in_features)
            ),
        )
        parametrize.register_parametrization(
            self, "weight", ortho_params.orthogonalizer(weight_shape=self.weight.shape)
        )

    def singular_values(self):
        svs = np.linalg.svd(
            self.weight.detach().cpu().numpy(), full_matrices=False, compute_uv=False
        )
        stable_rank = np.sum(np.mean(svs)) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank


class UnitNormLinear(nn.Linear):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """LInear layer where each output unit is normalized to have Frobenius norm 1"""
        super(UnitNormLinear, self).__init__(*args, **kwargs)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            L2Normalize(dtype=self.weight.dtype, dim=1),
        )

    def singular_values(self):
        svs = np.linalg.svd(
            self.weight.detach().cpu().numpy(), full_matrices=False, compute_uv=False
        )
        stable_rank = np.sum(np.mean(svs)) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank
