from math import ceil
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_1_t
from torch.nn.common_types import _size_2_t
from torch.nn.common_types import _size_3_t


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


class L2Normalize(nn.Module):
    def __init__(self, dim=(2, 3)):
        super(L2Normalize, self).__init__()
        self.dim = dim

    def forward(self, kernel):
        norm = torch.sqrt(torch.sum(kernel**2, dim=self.dim, keepdim=True))
        return kernel / (norm + 1e-12)

    def right_inverse(self, kernel):
        # we assume that the kernel is normalized
        norm = torch.sqrt(torch.sum(kernel**2, dim=self.dim, keepdim=True))
        return kernel / (norm + 1e-12)


class BatchedPowerIteration(nn.Module):
    def __init__(self, kernel_shape, power_it_niter=3, eps=1e-12):
        """
        This module is a batched version of the Power Iteration algorithm.
        It is used to normalize the kernel of a convolutional layer.

        Args:
            kernel_shape (tuple): shape of the kernel, the last dimension will be normalized.
            power_it_niter (int, optional): number of iterations. Defaults to 3.
            eps (float, optional): small value to avoid division by zero. Defaults to 1e-12.
        """
        super(BatchedPowerIteration, self).__init__()
        self.kernel_shape = kernel_shape
        self.power_it_niter = power_it_niter
        self.eps = eps
        # init u
        # u will be kernel_shape[:-2] + (kernel_shape[:-2], 1)
        # v will be kernel_shape[:-2] + (kernel_shape[:-1], 1,)
        self.u = nn.Parameter(
            torch.Tensor(torch.randn(*kernel_shape[:-2], kernel_shape[-2], 1)),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.Tensor(torch.randn(*kernel_shape[:-2], kernel_shape[-1], 1)),
            requires_grad=False,
        )
        parametrize.register_parametrization(self, "u", L2Normalize(dim=(-2)))
        parametrize.register_parametrization(self, "v", L2Normalize(dim=(-2)))

    def forward(self, X, init_u=None, n_iters=3, return_uv=True):
        for _ in range(n_iters):
            self.v = X.transpose(-1, -2) @ self.u
            self.u = X @ self.v
        # stop gradient on u and v
        u = self.u.detach()
        v = self.v.detach()
        # but keep gradient on s
        s = u.transpose(-1, -2) @ X @ v
        return X / (s + self.eps)

    def right_inverse(self, normalized_kernel):
        # we assume that the kernel is normalized
        return normalized_kernel


class BatchedBjorckOrthogonalization(nn.Module):
    def __init__(self, weight_shape, beta=0.5, backprop_iters=3, non_backprop_iters=10):
        self.weight_shape = weight_shape
        self.beta = beta
        self.backprop_iters = backprop_iters
        self.non_backprop_iters = non_backprop_iters
        if weight_shape[-2] < weight_shape[-1]:
            self.wwtw_op = BatchedBjorckOrthogonalization.wwt_w_op
        else:
            self.wwtw_op = BatchedBjorckOrthogonalization.w_wtw_op
        super(BatchedBjorckOrthogonalization, self).__init__()

    @staticmethod
    def w_wtw_op(w):
        return w @ (w.transpose(-1, -2) @ w)

    @staticmethod
    def wwt_w_op(w):
        return (w @ w.transpose(-1, -2)) @ w

    def forward(self, w):
        for _ in range(self.backprop_iters):
            w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
            # w_t_w = w.transpose(-1, -2) @ w
            # w = (1 + self.beta) * w - self.beta * w @ w_t_w
        w = GradPassthroughBjorck.apply(
            w, self.beta, self.non_backprop_iters, self.wwtw_op
        )
        return w

    def right_inverse(self, w):
        return w


class GradPassthroughBjorck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, beta, iters, wwtw_op):
        for _ in range(iters):
            w = (1 + beta) * w - beta * wwtw_op(w)
            # w_t_w = w.transpose(-1, -2) @ w
            # w = (1 + self.beta) * w - self.beta * w @ w_t_w
        return w

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass logic goes here.
        # You can retrieve saved variables using saved_tensors = ctx.saved_tensors.
        grad_input = (
            grad_output.clone()
        )  # For illustration, this just passes the gradient through.
        return grad_input, None, None, None


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
        bjorck_nbp_iters=0,
        bjorck_bp_iters=25,
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
            padding = kernel_size // 2 if kernel_size != stride else 0
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

        if (self.in_channels != self.inner_channels) or (self.kernel_size == 1):
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
        if self.kernel_size != self.stride:
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
        if (self.out_channels != self.inner_channels) or (stride > 1):
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
        self.weight.requires_grad = False
        # buffer to store cached weight
        with torch.no_grad():
            # assign the weight to the cached weight
            self.weight.data = FlashBCOP.merge_kernels(
                self.preconv_weight,
                self.mainconv_weight,
                self.postconv_weight,
                self.in_channels,
                self.out_channels,
                self.inner_channels,
                self.stride,
                self.groups,
            )
        # self.register_buffer("weight_buffer", self.weight)

    def singular_values(self):
        if self.padding != "circular":
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
            if self.preconv_weight is not None:
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
            if self.mainconv_weight is not None:
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
            if self.postconv_weight is not None:
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

    @staticmethod
    def merge_kernels(
        preconv_weight,
        mainconv_weight,
        postconv_weight,
        cin,
        cout,
        min_c,
        stride,
        groups,
    ):
        if preconv_weight is not None:
            preconv_weight = preconv_weight.view(min_c, cin // groups, 1, 1)
        if postconv_weight is not None:
            # reshape from (groups, cout // groups, min_c // groups * stride * stride)
            # to (cout, min_c // groups, stride, stride)
            postconv_weight = postconv_weight.view(
                cout, min_c // groups, stride, stride
            )
        if cin <= min_c:
            return fast_matrix_conv(
                fast_matrix_conv(preconv_weight, mainconv_weight, groups),
                postconv_weight,
                groups,
            )
        else:
            return fast_matrix_conv(
                preconv_weight,
                fast_matrix_conv(mainconv_weight, postconv_weight, groups),
                groups,
            )

    def forward(self, x):
        self._input_shape = x.shape[2:]
        if self.training:
            weight = FlashBCOP.merge_kernels(
                self.preconv_weight,
                self.mainconv_weight,
                self.postconv_weight,
                self.in_channels,
                self.out_channels,
                self.inner_channels,
                self.stride,
                self.groups,
            )
            self.weight.data = weight.data
        else:
            weight = self.weight.data
        return self._conv_forward(x, weight, self.bias)
