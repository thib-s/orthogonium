from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time
import einops
import torch.nn.utils.parametrize as parametrize


class Skew(nn.Module):
    def forward(self, kernel):
        kernel_t = torch.transpose(kernel, 0, 1)
        kernel_t = torch.flip(kernel_t, [2, 3])
        return kernel - kernel_t

    def right_inverse(self, kernel):
        return kernel


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


class PowerIterationConv(nn.Module):
    def __init__(self, in_channels, kernel_size, power_it_niter=3):
        super(PowerIterationConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.power_it_niter = power_it_niter
        # init u
        self.u = nn.Parameter(
            torch.Tensor(
                torch.randn(
                    1,
                    self.in_channels,
                    2 * self.kernel_size + 1,
                    2 * self.kernel_size + 1,
                )
            ),
            requires_grad=False,
        )
        parametrize.register_parametrization(self, "u", L2Normalize())

    def forward(self, kernel):
        # TODO: when cin != cout
        # we should have conv(conv_transpose(x))
        # with the appropriate order
        for i in range(self.power_it_niter):
            # since we are doing circular padding, we can use u whose shape is 2*k+1
            u2 = F.conv2d(
                F.pad(
                    self.u.detach(),
                    (
                        self.kernel_size // 2,
                        self.kernel_size // 2,
                        self.kernel_size // 2,
                        self.kernel_size // 2,
                    ),
                    mode="circular",
                ),
                kernel,
            )
            self.u = u2  # assignment will normalize self.u
        sigmas = torch.norm(u2, dim=(2, 3), keepdim=True)
        self.u.data = u2
        return kernel / sigmas.transpose(0, 1)

    def right_inverse(self, normalized_kernel):
        # we assume that the kernel is normalized
        return normalized_kernel


class FantasticFour(nn.Module):
    def __init__(self, in_channels, kernel_shape, power_it_niter=3):
        super(FantasticFour, self).__init__()
        self.in_channels = in_channels
        self.kernel_shape = kernel_shape
        self.power_it_niter = power_it_niter
        # init u
        out_ch, in_ch, h, w = kernel_shape
        self.u1 = nn.Parameter(torch.randn((1, in_ch, 1, w)), requires_grad=False)
        parametrize.register_parametrization(self, "u1", L2Normalize(dim=(0, 1, 2, 3)))
        self.u2 = nn.Parameter(torch.randn((1, in_ch, h, 1)), requires_grad=False)
        parametrize.register_parametrization(self, "u2", L2Normalize(dim=(0, 1, 2, 3)))
        self.u3 = nn.Parameter(torch.randn((1, in_ch, h, w)), requires_grad=False)
        parametrize.register_parametrization(self, "u3", L2Normalize(dim=(0, 1, 2, 3)))
        self.u4 = nn.Parameter(torch.randn((out_ch, 1, h, w)), requires_grad=False)
        parametrize.register_parametrization(self, "u4", L2Normalize(dim=(0, 1, 2, 3)))
        self.v1 = nn.Parameter(torch.randn((out_ch, 1, h, 1)), requires_grad=False)
        parametrize.register_parametrization(self, "v1", L2Normalize(dim=(0, 1, 2, 3)))
        self.v2 = nn.Parameter(torch.randn((out_ch, 1, 1, w)), requires_grad=False)
        parametrize.register_parametrization(self, "v2", L2Normalize(dim=(0, 1, 2, 3)))
        self.v3 = nn.Parameter(torch.randn((out_ch, 1, 1, 1)), requires_grad=False)
        parametrize.register_parametrization(self, "v3", L2Normalize(dim=(0, 1, 2, 3)))
        self.v4 = nn.Parameter(torch.randn((1, in_ch, 1, 1)), requires_grad=False)
        parametrize.register_parametrization(self, "v4", L2Normalize(dim=(0, 1, 2, 3)))

    def forward(self, kernel):
        for i in range(self.power_it_niter):
            self.v1 = (kernel * self.u1).sum((1, 3), keepdim=True)
            self.u1 = (kernel * self.v1).sum((0, 2), keepdim=True)
            self.v2 = (kernel * self.u2).sum((1, 2), keepdim=True)
            self.u2 = (kernel * self.v2).sum((0, 3), keepdim=True)
            self.v3 = (kernel * self.u3).sum((1, 2, 3), keepdim=True)
            self.u3 = (kernel * self.v3).sum(0, keepdim=True)
            self.v4 = (kernel * self.u4.data).sum((0, 2, 3), keepdim=True)
            self.u4 = (kernel * self.v4).sum(1, keepdim=True)
        sigma1 = torch.sum(kernel * self.u1.detach() * self.v1.detach())
        sigma2 = torch.sum(kernel * self.u2.detach() * self.v2.detach())
        sigma3 = torch.sum(kernel * self.u3.detach() * self.v3.detach())
        sigma4 = torch.sum(kernel * self.u4.detach() * self.v4.detach())
        sigma = torch.min(torch.min(torch.min(sigma1, sigma2), sigma3), sigma4)
        return kernel / sigma

    def right_inverse(self, normalized_kernel):
        # we assume that the kernel is normalized
        return normalized_kernel


class ConvExponential(nn.Module):
    def __init__(self, in_channels, kernel_size, exp_niter=5):
        super(ConvExponential, self).__init__()
        self.in_channels = in_channels  # assuming that cin == cout
        self.kernel_size = kernel_size
        self.exp_niter = exp_niter
        self.pad_fct = nn.ConstantPad2d((self.kernel_size - 1) // 2, 0)
        # build the identity kernel
        identity_kernel = torch.eye(in_channels, in_channels).view(
            [in_channels, in_channels, 1, 1]
        )
        identity_kernel = self.pad_fct(identity_kernel)  # pad to kernel size
        self.register_buffer("identity_kernel", identity_kernel)

    def forward(self, kernel):
        # init kernel_global with the two first terms of the exponential
        kernel_global = kernel + self.identity_kernel
        kernel_global = kernel_global.unsqueeze(0)
        kernel_i = kernel.unsqueeze(0)
        conv_filter_n_perm = kernel.permute(1, 0, 2, 3).unsqueeze(2)
        # compute each terms after the 2nd and aggregate to kernel_global
        for i in range(2, self.exp_niter + 1):
            # this code is equivalent to fast_dw_matrix_conv
            # without the unnecessary permutations
            # m1 is 1, m, n, k1, k2
            # m2 is m, n, 1, l1, l2
            # Run conv, output shape 1*m*n*(k+l-1)*(k+l-1)
            kernel_i = F.conv_transpose3d(kernel_i, conv_filter_n_perm) / float(i)
            # aggregate to kernel_global
            kernel_global = self.pad_fct(kernel_global) + kernel_i
        # remove extra dims used to for the fast matrix conv trick
        return kernel_global.squeeze(0)


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    return np.linalg.svd(transforms, compute_uv=False)


def fast_matrix_conv(m1, m2):
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
    r2 = r2.squeeze(0)  # (k+l-1)*(k+l-1)*n*nb
    return r2


class SOC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding_type="circular",
        bias=True,
        exp_niter=5,
        power_it_niter=3,
    ):
        super(SOC, self).__init__()
        assert (stride == 1) or (stride == 2)
        self.out_channels = out_channels
        self.in_channels = in_channels * stride * stride
        self.max_channels = max(self.out_channels, self.in_channels)
        self.padding_type = padding_type
        self.stride = stride
        self.kernel_size = kernel_size
        self.exp_niter = exp_niter
        self.power_it_niter = power_it_niter

        self.weight = nn.Parameter(
            torch.Tensor(
                torch.randn(
                    self.max_channels,
                    self.max_channels,
                    self.kernel_size,
                    self.kernel_size,
                )
            ),
            requires_grad=True,
        )
        parametrize.register_parametrization(self, "weight", Skew())
        parametrize.register_parametrization(
            self,
            "weight",
            FantasticFour(
                self.max_channels, self.weight.shape, power_it_niter=self.power_it_niter
            ),
        )
        parametrize.register_parametrization(
            self,
            "weight",
            ConvExponential(
                self.max_channels, self.kernel_size, exp_niter=self.exp_niter
            ),
            unsafe=True,
        )
        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels), requires_grad=True
            )
        else:
            self.bias = None

    def singular_values(self):
        # Implements interface required by LipschitzModuleL2
        svs = torch.from_numpy(
            conv_singular_values_numpy(
                self.weight.detach().cpu().numpy(), self._input_shape
            )
        ).to(device=self.weight.device)
        return svs

    def forward(self, x):
        self._input_shape = x.shape[2:]

        if self.stride > 1:
            x = einops.rearrange(
                x,
                "b c (w k1) (h k2) -> b (c k1 k2) w h",
                k1=self.stride,
                k2=self.stride,
            )

        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            z = F.pad(x, p4d)
        else:
            z = x

        with parametrize.cached():
            # do it with circular padding
            if self.padding_type not in ["same", "valid"]:
                kw, kh = self.weight.shape[2:]
                z = F.pad(
                    z, (kw // 2, kh // 2, kw // 2, kh // 2), mode=self.padding_type
                )
            z = F.conv2d(z, self.weight, padding=self.padding_type)

        if self.out_channels < self.in_channels:
            z = z[:, : self.out_channels, :, :]

        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z
