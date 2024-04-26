from typing import Union

from torch import nn as nn
from torch.nn.common_types import _size_2_t

from flashlipschitz.layers.conv.bcop_x_rko_conv import BcopRkoConv2d
from flashlipschitz.layers.conv.fast_block_ortho_conv import (
    FlashBCOP,
)
from flashlipschitz.layers.conv.reparametrizers import BjorckParams
from flashlipschitz.layers.conv.rko_conv import RKOConv2d


def OrthoConv2d(
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
) -> nn.Conv2d:
    """
    factory function to create an Orthogonal Convolutional layer
    choosing the appropriate class depending on the kernel size and stride
    """
    if kernel_size < stride:
        raise RuntimeError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        return RKOConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            bjorck_params,
        )
    elif stride == 1:
        return FlashBCOP(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            bjorck_params,
        )
    else:
        return BcopRkoConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            bjorck_params,
        )
