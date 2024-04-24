import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from flashlipschitz.layers.conv.reparametrizers import (
    BatchedBjorckOrthogonalization,
    L2Normalize,
    BjorckParams,
)
from flashlipschitz.layers.conv.reparametrizers import (
    BatchedPowerIteration,
)


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


class RKOParametrizer(nn.Module):
    def __init__(
        self,
        kernel_shape,
        groups,
        scale,
        bjorck_params: BjorckParams = BjorckParams(),
    ):
        super(RKOParametrizer, self).__init__()
        self.kernel_shape = kernel_shape
        self.groups = groups
        out_channels, in_channels, k1, k2 = kernel_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k1 = k1
        self.k2 = k2
        self.scale = scale
        self.register_module(
            "pi",
            BatchedPowerIteration(
                kernel_shape=(out_channels, in_channels * k1 * k2),
                power_it_niter=bjorck_params.power_it_niter,
                eps=bjorck_params.eps,
            ),
        )
        self.register_module(
            "bjorck",
            BatchedBjorckOrthogonalization(
                weight_shape=(out_channels, in_channels * k1 * k2),
                beta=bjorck_params.beta,
                niters=bjorck_params.bjorck_iters,
            ),
        )

    def forward(self, X):
        X = X.view(self.out_channels, self.in_channels * self.k1 * self.k2)
        X = self.pi(X)
        X = self.bjorck(X)
        X = X.view(self.out_channels, self.in_channels, self.k1, self.k2)
        return X / self.scale

    def right_inverse(self, X):
        return X


def attach_rko_weight(
    layer,
    weight_name,
    kernel_shape,
    groups,
    scale=None,
    bjorck_params: BjorckParams = BjorckParams(),
):
    out_channels, in_channels, kernel_size, k2 = kernel_shape
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
            bjorck_params,
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
        ortho_params: BjorckParams = BjorckParams(),
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
        self.scale = 1 / math.sqrt(kernel_size * kernel_size)
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
                **ortho_params.__dict__,
            ),
        )

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(RKOConv2d, self).forward(X)

    def singular_values(self):
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach().cpu().numpy(), self._input_shape
        )
        return sv_min, sv_max, stable_rank


class OrthoLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(OrthoLinear, self).__init__(in_features, out_features, bias=bias)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            BatchedPowerIteration((self.out_features, self.in_features)),
        )
        parametrize.register_parametrization(
            self, "weight", BatchedBjorckOrthogonalization(self.weight.shape)
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
            L2Normalize(dim=1),
        )

    def singular_values(self):
        svs = np.linalg.svd(
            self.weight.detach().cpu().numpy(), full_matrices=False, compute_uv=False
        )
        stable_rank = np.sum(np.mean(svs)) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank
