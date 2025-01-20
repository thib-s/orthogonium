import pytest
import torch
import numpy as np
from torch import nn

from orthogonium.layers import AdaptiveOrthoConv2d
from orthogonium.layers.conv.AOC import RKOConv2d
from orthogonium.layers.conv.AOL import AOLConv2D
from orthogonium.layers.conv.singular_values import get_conv_sv
from tests.test_orthogonality_conv import _compute_sv_impulse_response_layer


@pytest.fixture(scope="session")
def device():
    """Pytest fixture that returns 'cuda' if available, otherwise 'cpu'."""
    return "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "layer_cls,kwargs,img_shape",
    [
        # A basic Conv2d example
        (
            nn.Conv2d,
            dict(
                in_channels=3,
                out_channels=8,
                kernel_size=2,
                stride=1,
                padding=1,
                padding_mode="zeros",
            ),
            (3, 8, 8),  # (C, H, W)
        ),
        # Another Conv2d with groups>1
        (
            nn.Conv2d,
            dict(
                in_channels=8,
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
                groups=2,
            ),
            (8, 8, 8),
        ),
        (
            nn.Conv2d,
            dict(
                in_channels=6,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
                groups=1,
            ),
            (6, 8, 8),
        ),
        (
            AOLConv2D,
            dict(
                in_channels=6,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
            ),
            (6, 8, 8),
        ),
        (
            nn.Conv2d,
            dict(
                in_channels=6,
                out_channels=6,
                kernel_size=2,
                stride=2,
                padding=0,
                groups=1,
            ),
            (6, 8, 8),
        ),
        (
            AdaptiveOrthoConv2d,
            dict(
                in_channels=6,
                out_channels=6,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
                groups=1,
            ),
            (6, 8, 8),
        ),
        (
            AdaptiveOrthoConv2d,
            dict(
                in_channels=6,
                out_channels=6,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="circular",
                groups=1,
            ),
            (6, 8, 8),
        ),
        (
            nn.Conv2d,
            dict(
                in_channels=4,
                out_channels=16,
                kernel_size=2,
                stride=2,
                padding=0,
                groups=1,
            ),
            (4, 8, 8),
        ),
        (
            RKOConv2d,
            dict(
                in_channels=4,
                out_channels=2,
                kernel_size=2,
                stride=1,
                padding="same",
            ),
            (4, 8, 8),
        ),
    ],
)
def test_conv_sv_methods(layer_cls, kwargs, img_shape, device):
    """
    Parametrized test that checks whether get_conv_sv(...) and
    _compute_sv_impulse_response_layer(...) produce comparable results
    for various Conv2d (or subclass) configurations.
    """
    # fixing seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Create and move layer to the device
    layer = layer_cls(**kwargs, bias=False).to(device)

    # Delattre-based Lipschitz estimation
    # (adjust n_iter to your desired accuracy)
    lip_val, stab_rank = get_conv_sv(
        layer,
        n_iter=12 if kwargs.get("padding_mode") == "circular" else 4,
        agg_groups=True,
        device=device,
        detach=True,
        return_stab_rank=True,
        imsize=8,
    )

    # Impulse response-based singular value estimation
    # Note: We pass the shape as (C,H,W)
    min_sv, max_sv, ratio = _compute_sv_impulse_response_layer(layer, img_shape)

    print(f"Delattre: {lip_val:.6f}, Impulse: {max_sv:.6f}")
    print(f"Stab rank: {stab_rank:.6f}, Ratio: {ratio:.6f}")
    # Because these are approximate methods, allow some tolerance in comparison
    # Adjust rtol/atol according to your expected accuracy
    try:
        assert np.allclose(
            lip_val, max_sv, rtol=1e-1, atol=5e-2
        ), f"Lipschitz constant (Delattre) = {lip_val} not close to max SV (impulse) = {max_sv}"

        assert np.allclose(
            stab_rank, ratio, rtol=1e-1, atol=5e-2
        ), f"stable rank = {lip_val} not close to max SV (impulse) = {ratio}"
    except AssertionError as e:
        # test failed given the number of iterations, we have to rerun the test
        # with more iterations
        lip_val, stab_rank = get_conv_sv(
            layer,
            n_iter=15 if kwargs.get("padding_mode") == "circular" else 6,
            agg_groups=True,
            device=device,
            detach=True,
            return_stab_rank=True,
            imsize=8,
        )
        assert np.allclose(
            lip_val, max_sv, rtol=1e-1, atol=5e-2
        ), f"Lipschitz constant (Delattre) = {lip_val} not close to max SV (impulse) = {max_sv}"

        assert np.allclose(
            stab_rank, ratio, rtol=1e-1, atol=5e-2
        ), f"stable rank = {lip_val} not close to max SV (impulse) = {ratio}"
