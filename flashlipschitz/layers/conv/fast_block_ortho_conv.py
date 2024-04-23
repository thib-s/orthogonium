from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from flashlipschitz.layers.conv.reparametrizers import BatchedBjorckOrthogonalization
from flashlipschitz.layers.conv.reparametrizers import BatchedPowerIteration


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [0, 3, 4, 1, 2])  # g, k1, k2, ci, co
    transforms = np.fft.fft2(kernel, input_shape, axes=[1, 2])  # g, k1, k2, ci, co
    svs = np.linalg.svd(
        transforms, compute_uv=False, full_matrices=False
    )  # g, k1, k2, min(ci, co)
    stable_rank = np.mean(svs) / (svs.max() ** 2)
    return svs.min(), svs.max(), stable_rank


def fast_matrix_conv(m1, m2, groups=1):
    if m1 is None:
        return m2
    if m2 is None:
        return m1
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


def block_orth(p1, p2):
    assert p1.shape == p2.shape
    g, n, n2 = p1.shape
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    res = torch.einsum(
        "bgij,cgjk->gikbc", torch.stack([p1, eye - p1]), torch.stack([p2, eye - p2])
    )
    res = res.reshape(g * n, n, 2, 2)
    return res


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
        # we can rewrite PQ@PQ.t as an einsum
        PQ = torch.einsum("gijl,gikl->gijk", PQ, PQ)
        # PQ = PQ @ PQ.transpose(-1, -2)
        p = torch.eye(self.max_channels // self.groups) - 2 * PQ[:, 0]  # (g, c/g, c/g)
        p = p @ (torch.eye(self.max_channels // self.groups) - 2 * PQ[:, 1])  # (g,
        # c/g, c/g)
        p = p.view(self.max_channels, self.max_channels // self.groups, 1, 1)
        if self.in_channels != self.out_channels:
            p = p[:, : self.min_channels // self.groups, :, :]
        for _ in range(0, self.kernel_size - 1):
            p = fast_matrix_conv(
                p, block_orth(PQ[:, _ * 2], PQ[:, _ * 2 + 1]), self.groups
            )
        if self.transpose:
            # we do not perform flip since it does not affect orthogonality
            p = p.transpose(1, 2)
        # merge groups to get the final kernel
        p = p.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        return p


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
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_nbp_iters=5,
        bjorck_bp_iters=5,
    ):
        if (padding == "same") and (stride != 1):
            padding = (kernel_size - 1) // 2 if kernel_size != stride else 0

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
        self.num_kernels = 2 * self.kernel_size + 2
        self.groups = groups
        del self.weight
        self.weight = nn.Parameter(
            torch.Tensor(
                self.groups,
                self.num_kernels,
                self.max_channels // self.groups,
                self.max_channels // (self.groups * 2),
            ),
            requires_grad=True,
        )
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            BatchedPowerIteration(
                self.weight.shape,
                pi_iters,
            ),
        )
        parametrize.register_parametrization(
            self,
            "weight",
            BatchedBjorckOrthogonalization(
                self.weight.shape,
                bjorck_beta,
                bjorck_bp_iters,
                bjorck_nbp_iters,
            ),
        )
        parametrize.register_parametrization(
            self,
            "weight",
            BCOPTrivializer(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.groups,
            ),
            unsafe=True,
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
