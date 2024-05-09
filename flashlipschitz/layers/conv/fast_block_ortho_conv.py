import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from flashlipschitz.layers.conv.reparametrizers import (
    BatchedBjorckOrthogonalization,
    BjorckParams,
)
from flashlipschitz.layers.conv.reparametrizers import BatchedPowerIteration


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [0, 3, 4, 1, 2])  # g, k1, k2, ci, co
    transforms = np.fft.fft2(kernel, input_shape, axes=[1, 2])  # g, k1, k2, ci, co
    try:
        svs = np.linalg.svd(
            transforms, compute_uv=False, full_matrices=False
        )  # g, k1, k2, min(ci, co)
        stable_rank = np.mean(svs) / (svs.max() ** 2)
        return svs.min(), svs.max(), stable_rank
    except np.linalg.LinAlgError:
        print("numerical error in svd, returning only largest singular value")
        return None, np.linalg.norm(transforms, axis=(1, 2), ord=2), None


def fast_matrix_conv(m1, m2, groups=1):
    # m1 is m*n*k1*k2
    # m2 is nb*m*l1*l2
    m, n, k1, k2 = m1.shape
    nb, mb, l1, l2 = m2.shape
    assert m == mb * groups

    # Rearrange m1 for conv
    m1 = m1.transpose(0, 1)  # n*m*k1*k2

    # Rearrange m2 for conv
    m2 = m2.flip(-2, -1)

    # Run conv, output shape nb*n*(k+l-1)*(k+l-1)
    r2 = torch.nn.functional.conv2d(m1, m2, groups=groups, padding=(l1 - 1, l2 - 1))

    # Rearrange result
    return r2.transpose(0, 1)  # n*nb*(k+l-1)*(k+l-1)


def fast_batched_matrix_conv(m1, m2, groups=1):
    b, m, n, k1, k2 = m1.shape
    b2, nb, mb, l1, l2 = m2.shape
    assert m == mb * groups
    assert b == b2
    m1 = m1.view(b * m, n, k1, k2)
    m2 = m2.view(b * nb, mb, l1, l2)
    # Rearrange m1 for conv
    m1 = m1.transpose(0, 1)  # n*m*k1*k2
    # Rearrange m2 for conv
    m2 = m2.flip(-2, -1)
    r2 = torch.nn.functional.conv2d(m1, m2, groups=groups * b, padding=(l1 - 1, l2 - 1))
    # Rearrange result
    r2 = r2.view(n, b, nb, k1 + l1 - 1, k2 + l2 - 1)
    r2 = r2.permute(1, 2, 0, 3, 4)
    return r2


def block_orth(p1, p2, flip=False):
    assert p1.shape == p2.shape
    g, n, n2 = p1.shape
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    if flip:
        res = torch.einsum(
            "bgij,cgjk->gikbc", torch.stack([eye - p1, p1]), torch.stack([eye - p2, p2])
        )
    else:
        res = torch.einsum(
            "bgij,cgjk->gikbc", torch.stack([p1, eye - p1]), torch.stack([p2, eye - p2])
        )
    res = res.reshape(g * n, n, 2, 2)
    return res


def transpose_kernel(p, groups, flip=True):
    cig, cog, k1, k2 = p.shape
    cig = cig // groups
    # we do not perform flip since it does not affect orthogonality
    p = p.view(groups, cig, cog, k1, k2)
    p = p.transpose(1, 2)
    if flip:
        p = p.flip(-1, -2)
    # merge groups to get the final kernel
    p = p.reshape(cog * groups, cig, k1, k2)
    return p


class BCOPTrivializer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
    ):
        """This module is used to generate orthogonal kernels for the BCOP layer. It takes
        as input a matrix PQ of shape (groups, 2*kernel_size, c, c//2) and returns a kernel
        of shape (c, c, kernel_size, kernel_size) that is orthogonal.

        Args:
            kernel_size (int): size of the kernel.
            groups (int): number of groups in the convolution.
            has_projector (bool, optional): when set to True, PQ also include a projection
                matrix (i.e. a 1x1 convolution that allows to change the number of channels).
                Defaults to False.
            transpose (bool, optional): When set to True, the returned kernel is transposed.
                Defaults to False.
        """
        super(BCOPTrivializer, self).__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.min_channels = min(in_channels, out_channels)
        self.max_channels = max(in_channels, out_channels)
        self.transpose = out_channels < in_channels

    def forward(self, PQ):
        ident = torch.eye(self.max_channels // self.groups, device=PQ.device).unsqueeze(
            0
        )
        # we can rewrite PQ@PQ.t as an einsum
        PQ = torch.einsum("gijl,gikl->gijk", PQ, PQ)
        # PQ = PQ @ PQ.transpose(-1, -2)
        p = ident - 2 * PQ[:, 0, :, :]  # (g, c/g, c/g)
        p = p @ (ident - 2 * PQ[:, 1, :, :])  # (g, c/g, c/g)
        p = p.view(self.max_channels, self.max_channels // self.groups, 1, 1)
        if self.in_channels != self.out_channels:
            p = p[:, : self.min_channels // self.groups, :, :]
        for t2 in range(1, self.kernel_size):
            p = fast_matrix_conv(
                p,
                block_orth(PQ[:, t2 * 2], PQ[:, t2 * 2 + 1], flip=True),
                self.groups,
            )
        if self.transpose:
            p = transpose_kernel(p, self.groups, flip=False)
        return p.contiguous()


def attach_bcop_weight(
    layer,
    weight_name,
    kernel_shape,
    groups,
    bjorck_params: BjorckParams = BjorckParams(),
):
    out_channels, in_channels, kernel_size, k2 = kernel_shape
    in_channels *= groups
    assert kernel_size == k2, "only square kernels are supported for the moment"
    max_channels = max(in_channels, out_channels)
    num_kernels = 2 * kernel_size

    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(
            torch.Tensor(
                groups,
                num_kernels,
                max_channels // groups,
                max_channels // (groups * 2),
            ),
            requires_grad=True,
        ),
    )
    weight = getattr(layer, weight_name)
    torch.nn.init.orthogonal_(weight)
    parametrize.register_parametrization(
        layer,
        weight_name,
        BatchedPowerIteration(
            weight.shape,
            bjorck_params.power_it_niter,
        ),
    )
    parametrize.register_parametrization(
        layer,
        weight_name,
        BatchedBjorckOrthogonalization(
            weight.shape,
            bjorck_params.beta,
            bjorck_params.bjorck_iters,
        ),
    )
    parametrize.register_parametrization(
        layer,
        weight_name,
        BCOPTrivializer(
            in_channels,
            out_channels,
            kernel_size,
            groups,
        ),
        unsafe=True,
    )
    return weight


class FlashBCOP(nn.Conv2d):
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
        super(FlashBCOP, self).__init__(
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
        if (stride > 1) and (out_channels > in_channels):
            raise RuntimeError(
                "stride > 1 is not supported when out_channels > in_channels, "
                "use TODO layer instead"
            )
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
        attach_bcop_weight(
            self,
            "weight",
            (out_channels, in_channels // groups, kernel_size, kernel_size),
            groups,
            bjorck_params,
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
            self.weight.detach()
            .cpu()
            .view(
                self.groups,
                self.out_channels // self.groups,
                self.in_channels // self.groups,
                self.kernel_size,
                self.kernel_size,
            )
            .numpy(),
            self._input_shape,
        )
        return sv_min, sv_max, stable_rank

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(FlashBCOP, self).forward(X)


class BCOPTranspose(nn.ConvTranspose2d):
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
        super(BCOPTranspose, self).__init__(
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
        if out_channels * (stride**2) < in_channels:
            # raise warning because this configuration don't yield orthogonal
            # convolutions
            warnings.warn(
                "This configuration does not yield orthogonal convolutions due to "
                "padding issues: pytorch does not implement circular padding for "
                "transposed convolutions",
                RuntimeWarning,
            )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.groups = groups
        del self.weight
        attach_bcop_weight(
            self,
            "weight",
            (in_channels, out_channels // self.groups, kernel_size, kernel_size),
            groups,
            bjorck_params,
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
            self.weight.detach()
            .cpu()
            .reshape(
                self.groups,
                self.in_channels // self.groups,
                self.out_channels // self.groups,
                self.kernel_size,
                self.kernel_size,
            )
            .numpy(),
            self._input_shape,
        )
        return sv_min, sv_max, stable_rank

    def forward(self, X):
        self._input_shape = X.shape[2:]
        return super(BCOPTranspose, self).forward(X)
