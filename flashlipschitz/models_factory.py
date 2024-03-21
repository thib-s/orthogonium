import torch
import torch.nn as nn

from flashlipschitz.classparam import ClassParam
from flashlipschitz.layers import FlashBCOP
from flashlipschitz.layers import LayerCentering
from flashlipschitz.layers import MaxMin
from flashlipschitz.layers import OrthoLinear
from flashlipschitz.layers import ScaledAvgPool2d

ARCHS = {
    "PatchBasedExapandedCNN-S": dict(
        dim=128,
        depth=8,
        kernel_size=5,
        patch_size=2,
        expand_factor=2,
    ),
    "PatchBasedExapandedCNN-L": dict(
        dim=256,
        depth=8,
        kernel_size=5,
        patch_size=2,
        expand_factor=2,
    ),
    "PatchBasedExapandedCNN-XL": dict(
        dim=512,
        depth=8,
        kernel_size=5,
        patch_size=2,
        expand_factor=2,
    ),
}


def PatchBasedCNN(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
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
        act(),
        norm() if norm is not None else nn.Identity(),
        *[
            nn.Sequential(
                conv(in_channels=dim, out_channels=dim, kernel_size=kernel_size),
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


def PatchBasedExapandedCNN(
    img_shape=(3, 32, 32),
    dim=128,
    depth=8,
    kernel_size=3,
    patch_size=2,
    expand_factor=2,
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
                conv(
                    in_channels=dim,
                    out_channels=dim * expand_factor,
                    kernel_size=kernel_size,
                ),
                act(),
                norm() if norm is not None else nn.Identity(),
                conv(
                    in_channels=dim * expand_factor,
                    out_channels=dim,
                    kernel_size=kernel_size,
                ),
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
