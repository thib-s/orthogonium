from typing import Union
from torch import nn as nn
from torch.nn.common_types import _size_2_t
from orthogonium.layers.conv.AOC.rko_conv import RKOConv2d
from orthogonium.layers.conv.AOC.rko_conv import RkoConvTranspose2d
from orthogonium.layers.conv.adaptiveSOC.fast_skew_ortho_conv import (
    FastSOC,
    SOCTranspose,
)
from orthogonium.layers.conv.adaptiveSOC.soc_x_rko_conv import (
    SOCRkoConv2d,
    SOCRkoConvTranspose2d,
)
from orthogonium.reparametrizers import OrthoParams


def AdaptiveSOCConv2d(
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
) -> nn.Conv2d:
    """
    factory function to create an Orthogonal Convolutional layer
    choosing the appropriate class depending on the kernel size and stride.

    When kernel_size == stride, the layer is a RKOConv2d.
    When stride == 1, the layer is a FlashBCOP.
    Otherwise, the layer is a BcopRkoConv2d.
    """
    if kernel_size < stride:
        raise ValueError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RKOConv2d
    elif stride == 1:
        convclass = FastSOC
    else:
        convclass = SOCRkoConv2d
    return convclass(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        # ortho_params=ortho_params,
    )


def AdaptiveSOCConvTranspose2d(
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
) -> nn.ConvTranspose2d:
    """
    factory function to create an Orthogonal Convolutional Transpose layer
    choosing the appropriate class depending on the kernel size and stride.

    As we handle native striding with explicit kernel. It unlocks
    the possibility to use the same parametrization for transposed convolutions.
    This class uses the same interface as the ConvTranspose2d class.

    Unfortunately, circular padding is not supported for the transposed convolution.
    But unit testing have shown that the convolution is still orthogonal when
        `out_channels * (stride**2) > in_channels`.
    """
    if kernel_size < stride:
        raise ValueError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RkoConvTranspose2d
    elif stride == 1:
        convclass = SOCTranspose
    else:
        convclass = SOCRkoConvTranspose2d
    return convclass(
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
        # ortho_params=ortho_params,
    )
