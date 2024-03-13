import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time
from einops import rearrange
import numpy as np
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t
from typing import Optional


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    svs = np.linalg.svd(transforms, compute_uv=False, full_matrices=False)
    stable_rank = np.sum(np.mean(svs)) / (svs.max() ** 2)
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
    def __init__(self, num_kernels, cout, cin, power_it_niter=3, eps=1e-12):
        super(BatchedPowerIteration, self).__init__()
        self.num_kernels = num_kernels
        self.cin = cin
        self.cout = cout
        self.power_it_niter = power_it_niter
        self.eps = eps
        # init u
        if self.num_kernels is not None:
            self.u = nn.Parameter(
                torch.Tensor(torch.randn(num_kernels, cout, 1)),
                requires_grad=False,
            )
            self.v = nn.Parameter(
                torch.Tensor(torch.randn(num_kernels, cin, 1)),
                requires_grad=False,
            )
        else:
            self.u = nn.Parameter(
                torch.Tensor(torch.randn(cout, 1)),
                requires_grad=False,
            )
            self.v = nn.Parameter(
                torch.Tensor(torch.randn(cin, 1)),
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
            # w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
            w_t_w = w.transpose(-1, -2) @ w
            w = (1 + self.beta) * w - self.beta * w @ w_t_w
        w = GradPassthroughBjorck.apply(w, self.beta, self.non_backprop_iters)
        return w

    def right_inverse(self, w):
        return w


class GradPassthroughBjorck(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, beta, iters):
        for _ in range(iters):
            w_t_w = w.transpose(-1, -2) @ w
            w = (1 + beta) * w - beta * w @ w_t_w
        return w

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass logic goes here.
        # You can retrieve saved variables using saved_tensors = ctx.saved_tensors.
        grad_input = (
            grad_output.clone()
        )  # For illustration, this just passes the gradient through.
        return grad_input, None, None


def fast_matrix_conv(m1, m2):
    if m1 is None:
        return m2
    if m2 is None:
        return m1
    # m1 is m*n*k1*k2
    # m2 is nb*m*l1*l2
    m, n, k1, k2 = m1.shape
    nb, mb, l1, l2 = m2.shape
    assert m == mb

    # Rearrange m1 for conv
    # we put the n dimension in the depth dimension
    # and the m dimension in the channel dimension
    # to ensure NCDHW format
    m1 = m1.unsqueeze(0)  # 1*m*n*k1*k2

    # Rearrange m2 for conv
    # we transpose the spatial dimensions
    # and put the m dimension in the channel dimension
    # to ensure the kernel format: ic, oc, d, k1, k2
    # we expect the tensor to have shape mb*nb*1*l2*l1
    m2 = m2.transpose(0, 1)  # m*nb*l1*l2
    m2 = m2.unsqueeze(2)  # m*nb*1*l1*l2

    # Run conv, output shape 1*nb*n*(k+l-1)*(k+l-1)
    r2 = torch.nn.functional.conv_transpose3d(m1, m2)

    # Rearrange result
    r2 = r2.squeeze(0)  # nb*n*(k+l-1)*(k+l-1)
    return r2


def block_orth(p1, p2):
    assert p1.shape == p2.shape
    n = p1.size(0)
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    return torch.stack(
        [
            torch.stack([p1.mm(p2), p1.mm(eye - p2)]),
            torch.stack([(eye - p1).mm(p2), (eye - p1).mm(eye - p2)]),
        ]
    )


class ConvolutionOrthogonalGenerator(nn.Module):
    def __init__(
        self,
        kernel_size,
    ):
        super(ConvolutionOrthogonalGenerator, self).__init__()
        self.kernel_size = kernel_size

    @staticmethod
    def fast_matrix_conv(m1, m2):
        """modified version of fast_matrix_conv to avoid the transpose operation"""
        k1, k2, n, m = m1.shape
        l1, l2, mb, nb = m2.shape
        assert m == mb
        m1 = m1.unsqueeze(0)  # 1*k1*k2*n*m
        m1 = m1.permute(0, 4, 3, 1, 2)  # 1*n*k1*k1*m
        m2 = m2.permute(2, 3, 0, 1)  # m*nb*l1*l2
        m2 = m2.unsqueeze(2)  # m*nb*1*l1*l2
        r2 = F.conv_transpose3d(m1, m2)
        r2 = r2.squeeze(0).permute(2, 3, 1, 0)  # (k+l-1)*(k+l-1)*n*m
        return r2

    def forward(self, PQ):
        # we can rewrite PQ@PQ.t as an einsum
        # PQ = torch.einsum("ijl,ikl->ijk", PQ, PQ)
        PQ = PQ @ PQ.transpose(-1, -2)
        p = block_orth(PQ[0], PQ[1])
        for _ in range(0, self.kernel_size - 2):
            p = self.fast_matrix_conv(p, block_orth(PQ[_ * 2], PQ[_ * 2 + 1]))
        return p.permute(2, 3, 0, 1)

    def right_inverse(self, kernel):
        ci, co, k1, k2 = kernel.shape
        assert k1 == k2
        assert ci == co
        kernel = kernel.permute(2, 3, 0, 1)
        return kernel[:2, :, ci, : co // 2]


class BCOP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="circular",
        bias=True,
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_nbp_iters=25,
        bjorck_bp_iters=10,
        override_min_channels=None,
    ):
        """New parametrization of BCOP. It is in fact a sequence of 3 convolutions:
        - a 1x1 RKO convolution with in_channels inputs and min_channels outputs
            parametrized with RKO. It is orthogonal as it is a 1x1 convolution
        - a (k-s+1)x(k-s+1) BCOP conv with min_channels inputs and outputs
        - a sxs RKO convolution with stride s, min_channels inputs and out_channels outputs.

        Depending on the context (kernel size, stride, number of in/out channels) this method
        may only use 1 or 2 of the 3 described convolutions. Fusing the kernels result in a
        single kxk kernel with in_channels inputs and out_channels outputs with stride s.

        where min_channels = min(in_channels, out_channels) or overrided value
        when override_min_channels is not None.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): kernel size
            stride (int, optional): stride. Defaults to 1.
            padding (str, optional): padding type, can be any padding
                    handled by torch.nn.functional.pad, but most common are
                    "same", "valid" or "circular". Defaults to "circular".
            bias (bool, optional): enable bias. Defaults to True.
            pi_iters (int, optional): number of iterations used to normalize kernel. Defaults to 3.
            bjorck_beta (float, optional): beta factor used in Björk&Bowie algorithm. Must be
                    between 0 and 0.5. Defaults to 0.5.
            bjorck_nbp_iters (int, optional): number of iteration without backpropagation. Defaults to 25.
            bjorck_bp_iters (int, optional): number of iterations with backpropagation. Defaults to 10.
            override_min_channels (int, optional): allow to ovveride the number of channels in the
                    main convolution (which is set to min(in_channels, out_channels) by default).
                    It can be used to overparametrize the convolution (when set to greater values)
                    or to create rank deficient convolutions. Defaults to None.
        """
        super(BCOP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.min_channels = (
            override_min_channels
            if override_min_channels is not None
            else min(self.out_channels, self.in_channels)
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.mainconv_kernel_size = self.kernel_size - (stride - 1)
        self.num_kernels = 2 * self.mainconv_kernel_size
        self.bjorck_bp_iters = bjorck_bp_iters
        self.bjorck_nbp_iters = bjorck_nbp_iters
        self.bjorck_beta = bjorck_beta
        self.pi_iters = pi_iters

        if (self.in_channels != self.min_channels) and (
            self.kernel_size != self.stride
        ):
            # we have to add a preconvolution to reduce the number of channels
            self.preconv_weight = nn.Parameter(
                torch.Tensor(self.min_channels, self.in_channels),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.preconv_weight)
            parametrize.register_parametrization(
                self,
                "preconv_weight",
                BatchedPowerIteration(
                    None, self.min_channels, self.in_channels, self.pi_iters
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
                    self.num_kernels, self.min_channels, self.min_channels // 2
                ),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.mainconv_weight)
            parametrize.register_parametrization(
                self,
                "mainconv_weight",
                BatchedPowerIteration(
                    self.num_kernels,
                    self.min_channels,
                    self.min_channels // 2,
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
                ConvolutionOrthogonalGenerator(self.mainconv_kernel_size),
                unsafe=True,
            )
        else:
            self.mainconv_weight = None
        # declare the postconvolution
        if (self.out_channels != self.min_channels) or (stride > 1):
            self.postconv_weight = nn.Parameter(
                torch.Tensor(self.out_channels, self.min_channels, stride, stride),
                requires_grad=True,
            )
            torch.nn.init.orthogonal_(self.postconv_weight)
            parametrize.register_parametrization(
                self,
                "postconv_weight",
                RKO(
                    out_channels=out_channels,
                    in_channels=self.min_channels,
                    kernel_size=stride,
                    scale=1.0,
                    power_it_niter=pi_iters,
                    eps=1e-12,
                    beta=bjorck_beta,
                    backprop_iters=bjorck_bp_iters,
                    non_backprop_iters=bjorck_nbp_iters,
                ),
            )
        else:
            self.postconv_weight = None

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels), requires_grad=True
            )
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        # buffer to store cached weight
        with torch.no_grad():
            self.weight = BCOP.merge_kernels(
                self.preconv_weight,
                self.mainconv_weight,
                self.postconv_weight,
                self.in_channels,
                self.out_channels,
                self.min_channels,
            )
        self.register_buffer("weight_buffer", self.weight)

    def singular_values(self):
        if self.stride == 1:
            sv_min, sv_max, stable_rank = conv_singular_values_numpy(
                self.weight.detach().cpu().numpy(), self._input_shape
            )
        else:
            print("unable to compute full spectrum return min and max singular values")
            sv_min = 1
            sv_max = 1
            stable_ranks = []
            if self.preconv_weight is not None:
                svs = np.linalg.svd(
                    self.preconv_weight.detach().cpu().numpy(),
                    compute_uv=False,
                    full_matrices=True,
                )
                sv_min = sv_min * svs.min()
                sv_max = sv_max * svs.max()
                stable_ranks.append(np.sum(np.mean(svs)) / (svs.max() ** 2))
            if self.mainconv_weight is not None:
                sv_min, sv_max, s_r = conv_singular_values_numpy(
                    self.mainconv_weight.detach().cpu().numpy(), self._input_shape
                )
                sv_min = sv_min * sv_min
                sv_max = sv_max * sv_max
                stable_ranks.append(s_r)
            if self.postconv_weight is not None:
                svs = np.linalg.svd(
                    self.postconv_weight.view(
                        self.out_channels, self.min_channels * self.stride * self.stride
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
        preconv_weight, mainconv_weight, postconv_weight, cin, cout, min_c
    ):
        if preconv_weight is not None:
            preconv_weight = preconv_weight.view(min_c, cin, 1, 1)
        if cin <= min_c:
            return fast_matrix_conv(
                fast_matrix_conv(preconv_weight, mainconv_weight), postconv_weight
            )
        else:
            return fast_matrix_conv(
                preconv_weight, fast_matrix_conv(mainconv_weight, postconv_weight)
            )

    def forward(self, x):
        self._input_shape = x.shape[2:]
        if self.training:
            weight = BCOP.merge_kernels(
                self.preconv_weight,
                self.mainconv_weight,
                self.postconv_weight,
                self.in_channels,
                self.out_channels,
                self.min_channels,
            )
            self.weight = weight
        else:
            weight = self.weight
        # apply cyclic padding to the input and perform a standard convolution
        if self.padding not in [None, "same", "valid"]:
            x_pad = F.pad(
                x,
                (
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                    self.kernel_size // 2,
                ),
                mode=self.padding,
            )
            z = F.conv2d(x_pad, weight, padding="valid", stride=self.stride)
        else:
            z = F.conv2d(x, weight, padding=self.padding, stride=self.stride)
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z


class RKO(nn.Module):
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
        super(RKO, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.register_module(
            "pi",
            BatchedPowerIteration(
                num_kernels=None,
                cin=in_channels * kernel_size * kernel_size,
                cout=out_channels,
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
        X = X.reshape(
            self.out_channels, self.in_channels * self.kernel_size * self.kernel_size
        )
        X = self.pi(X)
        X = self.bjorck(X)
        X = X.reshape(
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
        pi_kwargs={},
        bjorck_kwargs={},
        scale=1.0,
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
        self.scale = scale / math.sqrt(kernel_size * kernel_size)
        parametrize.register_parametrization(
            self,
            "weight",
            RKO(
                out_channels,
                in_channels,
                kernel_size,
                self.scale,
                **pi_kwargs,
                **bjorck_kwargs,
            ),
        )

    def forward(self, X):
        # self._input_shape = X.shape[2:]
        return super(RKOConv2d, self).forward(X)

    def singular_values(self):
        # Implements interface required by LipschitzModuleL2
        sv_min, sv_max, stable_rank = conv_singular_values_numpy(
            self.weight.detach().cpu().numpy(), self._input_shape
        )
        return sv_min, sv_max, stable_rank


class OrthoLinear(nn.Linear):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(OrthoLinear, self).__init__(*args, **kwargs)
        torch.nn.init.orthogonal_(self.weight)
        parametrize.register_parametrization(
            self,
            "weight",
            BatchedPowerIteration(None, self.out_features, self.in_features),
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


class ScaledAvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: bool = None,
        k_coef_lip: float = 1.0,
    ):
        """
        Layer from the deel-torchlip project: https://github.com/deel-ai/deel-torchlip/blob/master/deel/torchlip/modules/pooling.py
        Average pooling operation for spatial data, but with a lipschitz bound.

        Args:
            kernel_size: The size of the window.
            stride: The stride of the window. Must be None or equal to
                ``kernel_size``. Default value is ``kernel_size``.
            padding: Implicit zero-padding to be added on both sides. Must
                be zero.
            ceil_mode: When True, will use ceil instead of floor to compute the output
                shape.
            count_include_pad: When True, will include the zero-padding in the averaging
                calculation.
            divisor_override: If specified, it will be used as divisor, otherwise
                ``kernel_size`` will be used.
            k_coef_lip: The Lipschitz factor to ensure. The output will be scaled
                by this factor.

        This documentation reuse the body of the original torch.nn.AveragePooling2D
        doc.
        """
        torch.nn.AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
        if isinstance(kernel_size, tuple):
            self.scalingFactor = math.sqrt(np.prod(np.asarray(kernel_size)))
        else:
            self.scalingFactor = kernel_size

        if self.stride != self.kernel_size:
            raise RuntimeError("stride must be equal to kernel_size.")
        if np.sum(self.padding) != 0:
            raise RuntimeError(f"{type(self)} does not support padding.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.AvgPool2d.forward(self, input) * self.scalingFactor
