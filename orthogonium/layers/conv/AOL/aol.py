import torch
from torch import nn
from torch.nn.utils import parametrize

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.SLL.sll_layer import safe_inv


class AOLReparametrizer(nn.Module):
    def __init__(self, nb_features, groups):
        super(AOLReparametrizer, self).__init__()
        self.nb_features = nb_features
        self.groups = groups
        self.q = nn.Parameter(torch.randn(nb_features))

    def forward(self, kernel):
        ktk = nn.functional.conv2d(
            kernel,
            kernel,
            groups=1,
            padding=kernel.shape[-1] - 1,
        )
        ktk = torch.abs(ktk)
        q = torch.exp(self.q).reshape(-1, 1, 1, 1)
        q_inv = torch.exp(-self.q).reshape(-1, 1, 1, 1)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(torch.sqrt(t))
        t = t.reshape(-1, 1, 1, 1)
        kernel = kernel * t
        return kernel


class AOLConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        parametrize.register_parametrization(
            self,
            "weight",
            AOLReparametrizer(
                out_channels,
                groups=groups,
            ),
        )


class AOLConvTranspose2D(nn.ConvTranspose2d):
    """
    Example transposed convolution layer that registers the same
    AOLReparametrizer as in your AOLConv2D layer, but adapted to
    nn.ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # Register the same AOLReparametrizer
        parametrize.register_parametrization(
            self,
            "weight",
            AOLReparametrizer(in_channels, groups=groups),
        )
