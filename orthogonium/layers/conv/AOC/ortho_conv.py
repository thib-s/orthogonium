from typing import Union

from torch import nn as nn
from torch.nn.common_types import _size_2_t

from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConv2d
from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConvTranspose2d
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import FastBlockConvTranspose2D
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import FastBlockConv2d
from orthogonium.layers.conv.AOC.rko_conv import RKOConv2d
from orthogonium.layers.conv.AOC.rko_conv import RkoConvTranspose2d
from orthogonium.reparametrizers import OrthoParams


def AdaptiveOrthoConv2d(
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

    - When kernel_size == stride, the layer is a RKOConv2d.
    - When stride == 1, the layer is a FlashBCOP.
    - Otherwise, the layer is a BcopRkoConv2d.

    """
    if kernel_size < stride:
        raise RuntimeError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RKOConv2d
    elif (stride == 1) or (in_channels >= out_channels):
        convclass = FastBlockConv2d
    else:
        convclass = BcopRkoConv2d
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
        ortho_params=ortho_params,
    )


def AdaptiveOrthoConvTranspose2d(
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
        raise RuntimeError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RkoConvTranspose2d
    elif stride == 1:
        convclass = FastBlockConvTranspose2D
    else:
        convclass = BcopRkoConvTranspose2d
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
        ortho_params=ortho_params,
    )
