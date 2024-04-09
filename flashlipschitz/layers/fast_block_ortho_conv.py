from math import ceil
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_1_t
from torch.nn.common_types import _size_2_t
from torch.nn.common_types import _size_3_t

from flashlipschitz.layers.reparametrizers import BatchedBjorckOrthogonalization
from flashlipschitz.layers.reparametrizers import BatchedPowerIteration


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
        kernel_size,
        groups,
        has_projector=False,
        transpose=False,
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
        self.has_projector = has_projector
        self.transpose = transpose

    def forward(self, PQ):
        # we can rewrite PQ@PQ.t as an einsum
        PQ = torch.einsum("gijl,gikl->gijk", PQ, PQ)
        # PQ = PQ @ PQ.transpose(-1, -2)
        p = block_orth(PQ[:, 0], PQ[:, 1])
        for _ in range(0, self.kernel_size - 2):
            p = fast_matrix_conv(
                p, block_orth(PQ[:, _ * 2], PQ[:, _ * 2 + 1]), self.groups
            )
        if self.has_projector:
            p = torch.einsum("gmnkl,gmn->gmnkl", p, PQ[:, -1])
        if self.transpose:
            # we do not perform flip since it does not affect orthogonality
            p = p.transpose(1, 2)
        return p


class FlashBCOP(nn.Module):
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
        override_inner_channels=None,
    ):
        """New parametrization of BCOP. It is in fact a sequence of 3 convolutions:
        - a 1x1 RKO convolution with in_channels inputs and inner_channels outputs
            parametrized with RKO. It is orthogonal as it is a 1x1 convolution
        - a (k-s+1)x(k-s+1) BCOP conv with inner_channels inputs and outputs
        - a sxs RKO convolution with stride s, inner_channels inputs and out_channels outputs.

        Depending on the context (kernel size, stride, number of in/out channels) this method
        may only use 1 or 2 of the 3 described convolutions. Fusing the kernels result in a
        single kxk kernel with in_channels inputs and out_channels outputs with stride s.

        where inner_channels = max(in_channels, out_channels) or overrided value
        when override_inner_channels is not None.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): kernel size
            stride (int, optional): stride. Defaults to 1.
            padding (str, optional): padding type, can be any padding
                    handled by torch.nn.functional.pad, but most common are
                    "same", "valid" or "circular". Defaults to "circular".
            bias (bool, optional): enable bias. Defaults to True.
            groups (int, optional): number of groups. Defaults to 1.
            pi_iters (int, optional): number of iterations used to normalize kernel. Defaults to 3.
            bjorck_beta (float, optional): beta factor used in Björk&Bowie algorithm. Must be
                    between 0 and 0.5. Defaults to 0.5.
            bjorck_nbp_iters (int, optional): number of iteration without backpropagation. Defaults to 25.
            bjorck_bp_iters (int, optional): number of iterations with backpropagation. Defaults to 10.
        """
        if (padding == "same") and (stride != 1):
            padding = (kernel_size - 1) // 2 if kernel_size != stride else 0
        super(FlashBCOP, self).__init__(
            # in_channels,
            # out_channels,
            # kernel_size,
            # stride,
            # padding,
            # dilation,
            # groups,
            # bias,
            # padding_mode,
        )
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        if override_inner_channels is not None:
            self.inner_channels = override_inner_channels
        else:
            if (stride**2) * in_channels >= out_channels:
                self.inner_channels = in_channels
            else:
                self.inner_channels = out_channels // (stride**2)

        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise RuntimeError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if bjorck_beta < 0 or bjorck_beta > 0.5:
            raise RuntimeError("bjorck_beta must be between 0 and 0.5")
        if (in_channels % groups != 0) and (out_channels % groups != 0):
            raise RuntimeError(
                "in_channels and out_channels must be divisible by groups"
            )
        if dilation != 1:
            raise RuntimeError("dilation not supported")
        if ((self.inner_channels // groups) < 2) and (kernel_size != stride):
            raise RuntimeError("inner conv must have at least 2 channels")
        if self.kernel_size % 2 == 0:
            if not ((stride == kernel_size) and (padding == 0 or padding == "valid")):
                raise RuntimeError(
                    "even kernel size only supported for stride == kernel_size and padding == 0 or 'valid'"
                )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.mainconv_kernel_size = self.kernel_size - (stride - 1)
        self.num_kernels = 2 * self.mainconv_kernel_size
        self.groups = groups
        self.bjorck_bp_iters = bjorck_bp_iters
        self.bjorck_nbp_iters = bjorck_nbp_iters
        self.bjorck_beta = bjorck_beta
        self.pi_iters = pi_iters
        # compute criteria to enable or disable each of the 3 subconvolutions
        self.preconv_enabled = (self.in_channels != self.inner_channels) or (
            self.kernel_size == 1
        )
        self.mainconv_enabled = self.kernel_size != self.stride
        self.postconv_enabled = (self.out_channels != self.inner_channels) or (
            stride > 1
        )
        # declare each of the 3 subconvolutions if needed
        if self.preconv_enabled:
            # we have to add a preconvolution to reduce the number of channels
            self.preconv_weight = nn.Parameter(
                torch.Tensor(
                    self.groups,
                    self.inner_channels // self.groups,
                    self.in_channels // self.groups,
                ),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.preconv_weight)
            parametrize.register_parametrization(
                self,
                "preconv_weight",
                BatchedPowerIteration(
                    self.preconv_weight.shape,
                    self.pi_iters,
                ),
            )
            parametrize.register_parametrization(
                self,
                "preconv_weight",
                BatchedBjorckOrthogonalization(
                    self.preconv_weight.shape,
                    self.bjorck_beta,
                    self.bjorck_bp_iters,
                    self.bjorck_nbp_iters,
                ),
            )
        else:
            self.preconv_weight = None
        # then declare the main convolution unconstrained weights
        if self.mainconv_enabled:
            self.mainconv_weight = nn.Parameter(
                torch.Tensor(
                    self.groups,
                    self.num_kernels,
                    self.inner_channels // self.groups,
                    self.inner_channels // (self.groups * 2),
                ),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.mainconv_weight)
            parametrize.register_parametrization(
                self,
                "mainconv_weight",
                BatchedPowerIteration(
                    self.mainconv_weight.shape,
                    self.pi_iters,
                ),
            )
            parametrize.register_parametrization(
                self,
                "mainconv_weight",
                BatchedBjorckOrthogonalization(
                    self.mainconv_weight.shape,
                    self.bjorck_beta,
                    self.bjorck_bp_iters,
                    self.bjorck_nbp_iters,
                ),
            )
            parametrize.register_parametrization(
                self,
                "mainconv_weight",
                BCOPTrivializer(self.mainconv_kernel_size, self.groups),
                unsafe=True,
            )
        else:
            self.mainconv_weight = None
        # declare the postconvolution
        if self.postconv_enabled:
            self.postconv_weight = nn.Parameter(
                torch.Tensor(
                    groups,
                    self.out_channels // groups,
                    (self.inner_channels * stride * stride) // groups,
                ),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.postconv_weight)
            parametrize.register_parametrization(
                self,
                "postconv_weight",
                BatchedPowerIteration(
                    self.postconv_weight.shape,
                    self.pi_iters,
                ),
            )
            parametrize.register_parametrization(
                self,
                "postconv_weight",
                BatchedBjorckOrthogonalization(
                    self.postconv_weight.shape,
                    self.bjorck_beta,
                    self.bjorck_bp_iters,
                    self.bjorck_nbp_iters,
                ),
            )
        else:
            self.postconv_weight = None

        # self.weight is a Parameter that is not trainable
        # self.weight.requires_grad = False
        with torch.no_grad():
            weight = self.merge_kernels()
            self.register_buffer("weight", weight)
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
        if self.stride == 1:
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
        else:
            print("unable to compute full spectrum return min and max singular values")
            sv_min = 1
            sv_max = 1
            stable_ranks = []
            if self.preconv_enabled:
                svs = np.linalg.svd(
                    self.preconv_weight.detach()
                    .cpu()
                    .reshape(
                        self.groups,
                        self.inner_channels // self.groups,
                        self.in_channels // self.groups,
                    )
                    .numpy(),
                    compute_uv=False,
                    full_matrices=True,
                )
                sv_min = sv_min * svs.min()
                sv_max = sv_max * svs.max()
                stable_ranks.append(np.sum(np.mean(svs)) / (svs.max() ** 2))
            if self.mainconv_enabled:
                sv_min, sv_max, s_r = conv_singular_values_numpy(
                    self.mainconv_weight.detach()
                    .cpu()
                    .reshape(
                        self.groups,
                        self.inner_channels // self.groups,
                        self.inner_channels // self.groups,
                        self.mainconv_kernel_size,
                        self.mainconv_kernel_size,
                    )
                    .numpy(),
                    self._input_shape,
                )
                sv_min = sv_min * sv_min
                sv_max = sv_max * sv_max
                stable_ranks.append(s_r)
            if self.postconv_enabled:
                svs = np.linalg.svd(
                    self.postconv_weight.view(
                        self.groups,
                        self.out_channels // self.groups,
                        (self.inner_channels * self.stride * self.stride)
                        // self.groups,
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    compute_uv=False,
                    full_matrices=False,
                )
                sv_min = sv_min * svs.min()
                sv_max = sv_max * svs.max()
                stable_ranks.append(np.sum(np.mean(svs)) / (svs.max() ** 2))
            stable_rank = np.mean(stable_ranks)
        return sv_min, sv_max, stable_rank

    def merge_kernels(
        self,
    ):
        if self.preconv_enabled:  # check is tensor is None
            preconv_weight = self.preconv_weight.view(
                self.inner_channels, self.in_channels // self.groups, 1, 1
            )
        else:
            preconv_weight = None
        if self.postconv_enabled:
            # reshape from (groups, cout // groups, min_c // groups * stride * stride)
            # to (cout, min_c // groups, stride, stride)
            postconv_weight = self.postconv_weight.view(
                self.out_channels,
                self.inner_channels // self.groups,
                self.stride,
                self.stride,
            )
        else:
            postconv_weight = None
        if self.mainconv_enabled:
            mainconv_weight = self.mainconv_weight
        else:
            mainconv_weight = None
        if self.in_channels <= self.inner_channels:
            return fast_matrix_conv(
                fast_matrix_conv(preconv_weight, mainconv_weight, self.groups),
                postconv_weight,
                self.groups,
            )
        else:
            return fast_matrix_conv(
                preconv_weight,
                fast_matrix_conv(mainconv_weight, postconv_weight, self.groups),
                self.groups,
            )

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return torch.nn.functional.conv2d(
                torch.nn.functional.pad(
                    input, (self.kernel_size // 2,) * 4, mode=self.padding_mode
                ),
                weight=weight,
                bias=bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
            )
        return torch.nn.functional.conv2d(
            input,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def forward(self, x):
        self._input_shape = x.shape[2:]
        # todo: somehow caching the weight is not working in distributed training context
        if self.training:
            weight = self.merge_kernels()
            self.weight = weight
        else:
            weight = self.weight
        return self._conv_forward(x, weight, self.bias)
