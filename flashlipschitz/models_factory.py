import torch
import torch.nn as nn

from flashlipschitz.classparam import ClassParam
from flashlipschitz.layers import FlashBCOP
from flashlipschitz.layers import GroupMix
from flashlipschitz.layers import HouseHolder
from flashlipschitz.layers import HouseHolder_Order_2
from flashlipschitz.layers import LayerCentering
from flashlipschitz.layers import MaxMin
from flashlipschitz.layers import OrthoLinear
from flashlipschitz.layers import ScaledAvgPool2d
from flashlipschitz.layers import UnitNormLinear
from flashlipschitz.layers.sll_layer import SDPBasedLipschitzConv


class ConcatResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.add_module("fn", fn)

    def forward(self, x):
        # split x
        x1, x2 = x.chunk(2, dim=1)
        # apply function
        out = self.fn(x2)
        # concat and return
        return torch.cat([x1, out], dim=1)


class Residual(nn.Module):
    def __init__(self, fn, init_val=0.0):
        super().__init__()
        self.add_module("fn", fn)
        self.alpha = nn.Parameter(torch.tensor(init_val), requires_grad=True)

    def forward(self, x):
        # split x
        # x1, x2 = x.chunk(2, dim=1)
        # apply function
        out = self.fn(x)
        # concat and return
        # return torch.cat([x1, out], dim=1)
        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1 - alpha) * out


def SplitConcatNet(
    img_shape=(3, 224, 224),
    n_classes=1000,
    expand_factor=2,
    block_depth=2,
    kernel_size=5,
    embedding_dim=1024,
    groups=16,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        padding_mode="zeros",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=8,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(UnitNormLinear, bias=False),
    norm=ClassParam(LayerCentering, dim=1),
):
    def resblock(in_channels, out_channels, n_blocks, conv, act, norm):
        layers = []
        if in_channels != out_channels:
            layers.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                )
            )
        # layers.append(act())
        layers.append(norm() if norm is not None else nn.Identity())
        for _ in range(n_blocks):
            layers.append(
                conv(
                    out_channels,
                    expand_factor * out_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                )
            )
            layers.append(norm() if norm is not None else nn.Identity())
            layers.append(act())
            # layers.append(
            #     conv(
            #         expand_factor * out_channels,
            #         out_channels,
            #         kernel_size=kernel_size,
            #         groups=groups,
            #     )
            # )
            if groups > 1:
                layers.append(
                    conv(expand_factor * out_channels, out_channels, kernel_size=1)
                )
                # layers.append(act())
                layers.append(norm() if norm is not None else nn.Identity())
        return layers

    layers = [
        conv(
            in_channels=img_shape[0],
            out_channels=embedding_dim // 8,
            kernel_size=7,
            stride=4,
            padding=7 // 2,
        ),
        act(),
        norm() if norm is not None else nn.Identity(),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        *resblock(embedding_dim // 8, embedding_dim // 8, block_depth, conv, act, norm),
        *resblock(embedding_dim // 8, embedding_dim // 4, block_depth, conv, act, norm),
        *resblock(embedding_dim // 4, embedding_dim // 2, block_depth, conv, act, norm),
        *resblock(embedding_dim // 2, embedding_dim, block_depth, conv, act, norm),
        nn.AvgPool2d(7, divisor_override=7),
        # nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        lin(embedding_dim, n_classes),
    ]
    return nn.Sequential(*layers)


SplitConcatNetConfigs = {
    "M": dict(
        expand_factor=4,
        block_depth=2,
        kernel_size=3,
        embedding_dim=2048,
        groups=32,
        conv=ClassParam(
            FlashBCOP,
            bias=False,
            kernel_size=3,
            padding="same",
            padding_mode="zeros",
            pi_iters=3,
            bjorck_beta=0.5,
            bjorck_bp_iters=8,
            bjorck_nbp_iters=0,
        ),
        act=ClassParam(MaxMin),
        lin=ClassParam(UnitNormLinear, bias=False),
        norm=ClassParam(LayerCentering, dim=1),
    ),
}


def LipResNet(
    img_shape=(3, 224, 224),
    n_classes=1000,
    stridedconv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        padding_mode="zeros",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    skipconv=ClassParam(
        SDPBasedLipschitzConv,
        inner_dim_factor=2,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear, bias=False),
    norm=ClassParam(LayerCentering, dim=-3),
):
    layers = [
        stridedconv(in_channels=img_shape[0], out_channels=64, kernel_size=7, stride=4),
        # act(),
        norm() if norm is not None else nn.Identity(),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        *ResNetBlock(64, 64, 3, stridedconv, skipconv, act, norm),
        *ResNetBlock(64, 128, 4, stridedconv, skipconv, act, norm),
        *ResNetBlock(128, 256, 6, stridedconv, skipconv, act, norm),
        *ResNetBlock(256, 512, 3, stridedconv, skipconv, act, norm),
        # nn.AvgPool2d(7, divisor_override=7),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        lin(512, n_classes),
    ]
    return nn.Sequential(*layers)


def ResNetBlock(in_channels, out_channels, n_blocks, stridedconvn, skipconv, act, norm):
    layers = []
    if in_channels != out_channels:
        layers.append(stridedconvn(in_channels, out_channels, kernel_size=3, stride=2))
    # layers.append(act())
    layers.append(norm() if norm is not None else nn.Identity())
    for _ in range(n_blocks):
        layers.append(skipconv(cin=out_channels))
    return layers


def LipVGG(
    img_shape=(3, 224, 224),
    n_classes=1000,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        padding_mode="zeros",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear, bias=False),
    norm=ClassParam(LayerCentering, dim=-3),
):
    layers = [
        conv(in_channels=img_shape[0], out_channels=64, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=128, out_channels=128, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=128, out_channels=256, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=256, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=256, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=256, out_channels=512, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=512, out_channels=512, kernel_size=3, stride=2),
        nn.Flatten(),  # fm = 7x7
        lin(512 * 7 * 7, 4096),
        act(),
        norm(dim=-1) if norm is not None else nn.Identity(),
        lin(4096, 4096),
        act(),
        norm(dim=-1) if norm is not None else nn.Identity(),
        lin(4096, n_classes),
    ]
    return nn.Sequential(*layers)


def PatchBasedCNN(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    n_classes=10,
    groups=1,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(nn.LPPool2d, norm_type=2),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering, dim=-3),
):
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            padding_mode="zeros",
        ),
        act(),
        norm() if norm is not None else nn.Identity(),
        *[
            nn.Sequential(
                conv(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    groups=groups,
                ),
                norm() if norm is not None else nn.Identity(),
                act(),
                (
                    GroupMix(g, dim // g)
                    if (g := (groups if i % 2 == 0 else dim // groups))
                    > 1  # number of group switch every layer
                    else nn.Identity()
                ),
            )
            for i in range(depth)
        ],
        # scaledAvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool(
            kernel_size=(img_shape[1] // patch_size, img_shape[2] // patch_size),
            stride=None,
        ),
        nn.Flatten(),
        lin(
            dim,
            n_classes,
        ),
    )


def PatchBasedExapandedCNN(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    expand_factor=2,
    groups=1,
    n_classes=10,
    skip=False,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(ScaledAvgPool2d),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering, dim=-3),
):
    if skip:
        skipco = Residual
    else:
        skipco = nn.Sequential
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        ),
        *[
            skipco(
                nn.Sequential(
                    conv(
                        in_channels=dim,
                        out_channels=dim * expand_factor,
                        kernel_size=kernel_size,
                        groups=groups,
                    ),
                    act(),
                    norm() if norm is not None else nn.Identity(),
                    # (
                    #     GroupMix(groups, dim * expand_factor // groups)
                    #     if groups > 1
                    #     else nn.Identity()
                    # ),
                    conv(
                        in_channels=dim * expand_factor,
                        out_channels=dim,
                        kernel_size=kernel_size,
                        groups=dim // groups,
                    ),
                    # GroupMix(dim // groups, groups) if groups > 1 else nn.Identity(),
                )
            )
            for i in range(depth)
        ],
        # scaledAvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool((img_shape[1] // patch_size, img_shape[2] // patch_size), None),
        nn.Flatten(),
        lin(
            dim,
            n_classes,
        ),
    )


def ConvMixerInspired(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    expand_factor=1,
    channels_per_group=1,
    n_classes=10,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    pool=ClassParam(ScaledAvgPool2d),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering, dim=-3),
):
    return nn.Sequential(
        conv(
            in_channels=img_shape[0],
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        ),
        *[
            nn.Sequential(
                # depthwise (with groups of channels_per_group channels)
                conv(
                    in_channels=dim,
                    out_channels=dim * expand_factor,
                    groups=(dim * expand_factor) // channels_per_group,
                    kernel_size=kernel_size,
                ),
                act(),
                norm() if norm is not None else nn.Identity(),
                # pointwise
                conv(
                    kernel_size=1,
                    in_channels=dim * expand_factor,
                    out_channels=dim,
                ),
                act(),
                norm() if norm is not None else nn.Identity(),
            )
            for i in range(depth)
        ],
        # scaledAvgPool2d is AvgPool2d but with a sqrt(w*h)
        # factor, as it would be 1/sqrt(w,h) lip otherwise
        pool((img_shape[1] // patch_size, img_shape[2] // patch_size), None),
        nn.Flatten(),
        lin(
            dim,
            n_classes,
        ),
    )


def BCOPLargeCNN(
    img_shape=(3, 32, 32),
    n_classes=10,
    conv=ClassParam(
        FlashBCOP,
        bias=True,
        padding="circular",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering, dim=-3),
):
    layers = [
        conv(in_channels=img_shape[0], out_channels=32, kernel_size=3, stride=1),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=32, out_channels=64, kernel_size=5, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        act(),
        norm() if norm is not None else nn.Identity(),
        conv(in_channels=64, out_channels=64, kernel_size=5, stride=2),
        act(),
        norm() if norm is not None else nn.Identity(),
        nn.Flatten(),
        lin((img_shape[1] // 4) * (img_shape[2] // 4) * 64, 512),
        act(),
        lin(512, 512),
        act(),
        lin(512, n_classes),
    ]

    return nn.Sequential(*layers)


def StagedCNN(
    img_shape=(3, 32, 32),
    dim_repeats=[(64, 2), (128, 2)],
    n_classes=10,
    conv=ClassParam(
        FlashBCOP,
        bias=False,
        padding="same",
        pi_iters=3,
        bjorck_beta=0.5,
        bjorck_bp_iters=10,
        bjorck_nbp_iters=0,
    ),
    act=ClassParam(MaxMin),
    lin=ClassParam(OrthoLinear),
    norm=ClassParam(LayerCentering, dim=-3),
):
    layers = []
    in_channels = img_shape[0]

    # Create convolutional blocks
    for dim, repeats in dim_repeats:
        # Add repeated conv layers
        for _ in range(repeats):
            layers.append(
                conv(in_channels=in_channels, out_channels=dim, kernel_size=3)
            )
            layers.append(act())
            layers.append(norm() if norm is not None else nn.Identity())
            in_channels = dim

        # Add strided convolution to separate blocks
        layers.append(
            conv(
                in_channels=in_channels,
                out_channels=in_channels * 2,
                kernel_size=3,
                stride=2,
            )
        )
        layers.append(act())
        layers.append(norm() if norm is not None else nn.Identity())
        in_channels *= 2

    # Flatten layer
    layers.append(nn.Flatten())

    # Add linear layers
    for dim, repeats in dim_repeats:
        for _ in range(repeats):
            layers.append(lin(in_channels, dim))
            layers.append(act())
            in_channels = dim

    # Final linear layer for classification
    layers.append(lin(in_channels, n_classes))

    return nn.Sequential(*layers)


MODELS = {
    "SplitConcatNet-M": SplitConcatNet,
    "LipResNet": LipResNet,
}
