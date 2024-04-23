import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from flashlipschitz.layers.conv.reparametrizers import (
    BatchedBjorckOrthogonalization, L2Normalize,
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


# dataclass with RKO params
class RKOParams:
    def __init__(
        self,
        power_it_niter=3,
        eps=1e-12,
        beta=0.5,
        backprop_iters=3,
        non_backprop_iters=10,
    ):
        self.power_it_niter = power_it_niter
        self.eps = eps
        self.beta = beta
        self.backprop_iters = backprop_iters
        self.non_backprop_iters = non_backprop_iters


class RKOParametrizer(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_size,
        scale,
        power_it_niter=3,
        eps=1e-12,
        beta=0.5,
        backprop_iters=3,
        non_backprop_iters=10,
    ):
        super(RKOParametrizer, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.register_module(
            "pi",
            BatchedPowerIteration(
                kernel_shape=(out_channels, in_channels * kernel_size * kernel_size),
                power_it_niter=power_it_niter,
                eps=eps,
            ),
        )
        self.register_module(
            "bjorck",
            BatchedBjorckOrthogonalization(
                weight_shape=(out_channels, in_channels * kernel_size * kernel_size),
                beta=beta,
                backprop_iters=backprop_iters,
                non_backprop_iters=non_backprop_iters,
            ),
        )

    def forward(self, X):
        X = X.view(
            self.out_channels, self.in_channels * self.kernel_size * self.kernel_size
        )
        X = self.pi(X)
        X = self.bjorck(X)
        X = X.view(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )
        return X / self.scale

    def right_inverse(self, X):
        return X


class RKOConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="valid",
        bias=True,
        rko_params=None,
    ):
        super(RKOConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        torch.nn.init.orthogonal_(self.weight)
        self.scale = 1 / math.sqrt(kernel_size * kernel_size)
        parametrize.register_parametrization(
            self,
            "weight",
            RKOParametrizer(
                out_channels,
                in_channels,
                kernel_size,
                self.scale,
                **rko_params.__dict__,
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
