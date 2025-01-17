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
    Factory function to create an orthogonal convolutional layer, selecting the appropriate class based on kernel
    size and stride. This is a modified implementation of the `Skew orthogonal convolution` [1], with significant
    modification from the original paper:


    - This implementation provide an explicit kernel (which is larger the original kernel size) so the forward is done
        in a single iteration. As described in [2].
    - This implementation avoid the use of channels padding to handle case where cin != cout. Similarly, stride is
        handled natively using the ad adaptive scheme.
    - the fantastic four method is replaced by AOL which allows to reduce the number of iterations required to
        converge.

    It aims to be more scalable to large networks and large image sizes, while enforcing orthogonality in the
    convolutional layers. This layer also intend to be compatible with all the feature of the `nn.Conv2d` class
    (e.g., striding, dilation, grouping, etc.). This method has an explicit kernel, which means that the forward
    operation is equivalent to a standard convolutional layer, but the weight are constrained to be orthogonal.

    Note:
        - this implementation changes the size of the kernel, which also change the padding semantics. Please adjust
            the padding according to the kernel size and the number of iterations.
        - current unit testing use a tolerance of 8e-2 sor this layer can be expected to be 1.08 lipschitz continuous.
            Similarly, the stable rank is evaluated loosely (must be greater than 0.5).

    Key Features:
    -------------
        - Enforces orthogonality, preserving gradient norms.
        - Supports native striding, dilation, grouped convolutions, and flexible padding.

    Behavior:
    -------------
        - When kernel_size == stride, the layer is an `RKOConv2d`.
        - When stride == 1, the layer is a `FastBlockConv2d`.
        - Otherwise, the layer is a `BcopRkoConv2d`.

    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_2_t): Size of the convolution kernel.
        stride (_size_2_t, optional): Stride of the convolution. Default is 1.
        padding (str or _size_2_t, optional): Padding mode or size. Default is "same".
        dilation (_size_2_t, optional): Dilation rate. Default is 1.
        groups (int, optional): Number of blocked connections from input to output channels. Default is 1.
        bias (bool, optional): Whether to include a learnable bias. Default is True.
        padding_mode (str, optional): Padding mode. Default is "circular".
        ortho_params (OrthoParams, optional): Parameters to control orthogonality. Default is `OrthoParams()`.

    Returns:
        A configured instance of `nn.Conv2d` (one of `RKOConv2d`, `FastBlockConv2d`, or `BcopRkoConv2d`).

    Raises:
        `ValueError`: If kernel_size < stride, as orthogonality cannot be enforced.


    References:
        - [1] Singla, S., & Feizi, S. (2021, July). Skew orthogonal convolutions. In International Conference
        on Machine Learning (pp. 9756-9766). PMLR.<https://arxiv.org/abs/2105.11417>
        - [2] Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
        An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
        <https://arxiv.org/abs/2501.07930>
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
    Factory function to create an orthogonal transposed convolutional layer, selecting the appropriate class based on
    kernel size and stride. This is a modified implementation of the `Skew orthogonal convolution` [1], with significant
    modification from the original paper:

    - This implementation provide an explicit kernel (which is larger the original kernel size) so the forward is done
        in a single iteration. As described in [2].
    - This implementation avoid the use of channels padding to handle case where cin != cout. Similarly, stride is
        handled natively using the ad adaptive scheme.
    - the fantastic four method is replaced by AOL which allows to reduce the number of iterations required to
        converge.

    It aims to be more scalable to large networks and large image sizes, while enforcing orthogonality in the
    convolutional layers. This layer also intend to be compatible with all the feature of the `nn.Conv2d` class
    (e.g., striding, dilation, grouping, etc.). This method has an explicit kernel, which means that the forward
    operation is equivalent to a standard convolutional layer, but the weight are constrained to be orthogonal.

    Note:
        - this implementation changes the size of the kernel, which also change the padding semantics. Please adjust
            the padding according to the kernel size and the number of iterations.
        - current unit testing use a tolerance of 8e-2 sor this layer can be expected to be 1.08 lipschitz continuous.
            Similarly, the stable rank is evaluated loosely (must be greater than 0.5).

    Key Features:
    -------------
        - Enforces orthogonality, preserving gradient norms.
        - Supports native striding, dilation, grouped convolutions, and flexible padding.

    Behavior:
    -------------
        - When kernel_size == stride, the layer is an `RKOConv2d`.
        - When stride == 1, the layer is a `FastBlockConv2d`.
        - Otherwise, the layer is a `BcopRkoConv2d`.

    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_2_t): Size of the convolution kernel.
        stride (_size_2_t, optional): Stride of the convolution. Default is 1.
        padding (str or _size_2_t, optional): Padding mode or size. Default is "same".
        dilation (_size_2_t, optional): Dilation rate. Default is 1.
        groups (int, optional): Number of blocked connections from input to output channels. Default is 1.
        bias (bool, optional): Whether to include a learnable bias. Default is True.
        padding_mode (str, optional): Padding mode. Default is "circular".
        ortho_params (OrthoParams, optional): Parameters to control orthogonality. Default is `OrthoParams()`.

    Returns:
        A configured instance of `nn.Conv2d` (one of `RKOConv2d`, `FastBlockConv2d`, or `BcopRkoConv2d`).

    Raises:
        `ValueError`: If kernel_size < stride, as orthogonality cannot be enforced.


    References:
        - [1] Singla, S., & Feizi, S. (2021, July). Skew orthogonal convolutions. In International Conference
        on Machine Learning (pp. 9756-9766). PMLR.<https://arxiv.org/abs/2105.11417>
        - [2] Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
        An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
        <https://arxiv.org/abs/2501.07930>
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
