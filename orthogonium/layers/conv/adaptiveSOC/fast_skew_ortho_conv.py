import warnings
from dataclasses import dataclass
from typing import Union
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import (
    fast_matrix_conv,
    transpose_kernel,
    conv_singular_values_numpy,
)
from orthogonium.layers.conv.AOL.aol import AOLReparametrizer
import math


@dataclass
class ExpParams:
    exp_niter: int = 3


class Skew(nn.Module):
    def __init__(self, groups):
        super(Skew, self).__init__()
        self.groups = groups

    def forward(self, kernel):
        return kernel - transpose_kernel(kernel, self.groups, flip=True)

    def right_inverse(self, kernel):
        return kernel


class ConvExponential(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, exp_niter=5):
        super(ConvExponential, self).__init__()
        self.in_channels = in_channels  # assuming that cin == cout
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.exp_niter = exp_niter
        self.pad_fct = nn.ConstantPad2d(
            (
                int(math.ceil((self.kernel_size - 1) / 2)),
                int(math.floor((self.kernel_size - 1) / 2)),
                int(math.ceil((self.kernel_size - 1) / 2)),
                int(math.floor((self.kernel_size - 1) / 2)),
            ),
            0,
        )
        max_channels = max(in_channels, out_channels)
        # build the identity kernel
        # start with g identity matrices of size c//g x c//g
        group_in_channels = max_channels // groups
        identity_kernel = torch.eye(group_in_channels).unsqueeze(-1).unsqueeze(-1)
        identity_kernel = identity_kernel.repeat(groups, 1, 1, 1, 1)
        identity_kernel = identity_kernel.view(max_channels, group_in_channels, 1, 1)
        identity_kernel = self.pad_fct(identity_kernel)  # pad to kernel size
        self.register_buffer("identity_kernel", identity_kernel)
        self.max_channels = max_channels

    def forward(self, kernel):
        # init kernel_global with the two first terms of the exponential
        kernel_global = kernel + self.identity_kernel
        kernel_i = kernel
        # compute each terms after the 2nd and aggregate to kernel_global
        for i in range(2, self.exp_niter + 1):
            kernel_i = fast_matrix_conv(kernel_i, kernel, groups=self.groups) / float(i)
            # aggregate to kernel_global
            kernel_global = self.pad_fct(kernel_global) + kernel_i
        # remove extra dims used to for the fast matrix conv trick
        # reshape and drop useless channels
        kernel_global = kernel_global.view(
            self.groups,
            self.max_channels // self.groups,
            self.max_channels // self.groups,
            kernel_global.shape[-2],
            kernel_global.shape[-1],
        )[
            :,
            : self.out_channels // self.groups,
            : self.in_channels // self.groups,
            :,
            :,
        ]
        return kernel_global.reshape(
            self.out_channels,
            self.in_channels // self.groups,
            kernel_global.shape[-2],
            kernel_global.shape[-1],
        )


def attach_soc_weight(
    layer, weight_name, kernel_shape, groups, exp_params: ExpParams = ExpParams()
):
    """
    Attach a weight to a layer and parametrize it with the BCOPTrivializer module.
    The attached weight will be the kernel of an orthogonal convolutional layer.

    Args:
        layer (torch.nn.Module): layer to which the weight will be attached
        weight_name (str): name of the weight
        kernel_shape (tuple): shape of the kernel (out_channels, in_channels/groups, kernel_size, kernel_size)
        groups (int): number of groups
        exp_params (ExpParams): parameters for the exponential algorithm.

    Returns:
        torch.Tensor: a handle to the attached weight
    """
    out_channels, in_channels, kernel_size, k2 = kernel_shape
    in_channels *= groups  # compute the real number of input channels
    assert (
        kernel_size == k2
    ), "only square kernels are supported (to compute skew symmetric kernels)"
    assert kernel_size % 2 == 1, "kernel size must be odd"
    max_channels = max(in_channels, out_channels)
    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(
            torch.Tensor(
                max_channels, max_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        ),
    )
    weight = getattr(layer, weight_name)
    torch.nn.init.orthogonal_(weight)
    parametrize.register_parametrization(
        layer,
        weight_name,
        Skew(groups),
        unsafe=False,
    )
    parametrize.register_parametrization(
        layer,
        weight_name,
        AOLReparametrizer(max_channels, groups=groups),
        unsafe=False,
    )
    parametrize.register_parametrization(
        layer,
        weight_name,
        ConvExponential(
            in_channels, out_channels, kernel_size, groups, exp_params.exp_niter
        ),
        unsafe=True,
    )
    return weight


class FastSOC(nn.Conv2d):
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
        exp_params: ExpParams = ExpParams(),
    ):
        """
        Fast implementation of the Block Circulant Orthogonal Parametrization (BCOP) for convolutional layers.
        This approach changes the original BCOP algorithm to make it more scalable and efficient. This implementation
        rewrite efficiently the block convolution operator as a single convolution operation. Also the iterative algorithm
        is parallelized in the associative scan fashion.

        This layer is a drop-in replacement for the nn.Conv2d layer. It is orthogonal and Lipschitz continuous while maintaining
        the same interface as the Con2d. Also this method has an explicit kernel, whihc allows to compute the singular values of
        the convolutional layer.

        Striding is not supported when out_channels > in_channels. Real striding is supported in BcopRkoConv2d. The use of
        OrthogonalConv2d is recommended.
        """
        super(FastSOC, self).__init__(
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
        self.max_channels = max(in_channels, out_channels)

        # raise runtime error if kernel size >= stride
        if ((stride > 1) and (out_channels > in_channels)) or (stride > kernel_size):
            raise ValueError(
                "stride > 1 is not supported when out_channels > in_channels, "
                "use TODO layer instead"
            )
        if kernel_size < stride:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        del self.weight
        attach_soc_weight(
            self,
            "weight",
            (out_channels, in_channels // groups, kernel_size, kernel_size),
            groups,
            exp_params=exp_params,
        )
        self.kernel_size = self.weight.shape[-2:]


class SOCTranspose(nn.ConvTranspose2d):
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
        exp_params: ExpParams = ExpParams(),
    ):
        """
        Extention of the BCOP algorithm to transposed convolutions. This implementation
        uses the same algorithm as the FlashBCOP layer, but the layer acts as a transposed
        convolutional layer.
        """
        super(SOCTranspose, self).__init__(
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
        self.max_channels = max(in_channels, out_channels)

        # raise runtime error if kernel size >= stride
        if kernel_size < stride:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if ((self.max_channels // groups) < 2) and (kernel_size != stride):
            raise ValueError("inner conv must have at least 2 channels")
        if out_channels * (stride**2) < in_channels:
            # raise warning because this configuration don't yield orthogonal
            # convolutions
            warnings.warn(
                "This configuration does not yield orthogonal convolutions due to "
                "padding issues: pytorch does not implement circular padding for "
                "transposed convolutions",
                RuntimeWarning,
            )
        del self.weight
        attach_soc_weight(
            self,
            "weight",
            (in_channels, out_channels // self.groups, kernel_size, kernel_size),
            groups,
            exp_params=exp_params,
        )
        self.kernel_size = self.weight.shape[-2:]
