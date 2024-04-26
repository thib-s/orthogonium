import numpy as np
import pytest
import torch

from flashlipschitz.layers.conv.ortho_conv import OrthoConvTranspose
from flashlipschitz.layers.conv.reparametrizers import BjorckParams


# from flashlipschitz.layers.conv.fast_block_ortho_conv import FlashBCOP

# from flashlipschitz.layers.conv.ortho_conv import OrthoConv as FlashBCOP


def _compute_sv_impulse_response_layer(layer, img_shape):
    with torch.no_grad():
        inputs = torch.eye(img_shape[0] * img_shape[1] * img_shape[2]).view(
            img_shape[0] * img_shape[1] * img_shape[2],
            img_shape[0],
            img_shape[1],
            img_shape[2],
        )
        outputs = layer(inputs)
        try:
            svs = torch.linalg.svdvals(outputs.view(outputs.shape[0], -1))
            return svs.min(), svs.max(), svs.mean() / svs.max()
        except np.linalg.LinAlgError:
            print("SVD failed returning only largest singular value")
            return torch.norm(outputs.view(outputs.shape[0], -1), p=2).max(), 0, 0


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [8, 16, 32])
@pytest.mark.parametrize("output_channels", [16, 32, 64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
def test_bcop(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0)
    padding_mode = "zeros"
    try:
        bcop = OrthoConvTranspose(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
            bjorck_params=BjorckParams(
                power_it_niter=3,
                eps=1e-6,
                beta=0.5,
                bjorck_iters=20,
            ),
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    imsize = 6
    # Test backpropagation and weight update
    try:
        bcop.train()
        opt = torch.optim.SGD(bcop.parameters(), lr=0.001)
        for i in range(25):
            opt.zero_grad()
            inp = torch.randn(1, input_channels, imsize, imsize)
            output = bcop(inp)
            loss = -output.mean()
            loss.backward()
            opt.step()
        bcop.eval()  # so impulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")

    # check that bcop.weight has the correct shape
    if bcop.weight.data.shape != (
        input_channels,
        output_channels // groups,
        kernel_size,
        kernel_size,
    ):
        pytest.fail(
            f"BCOP weight has incorrect shape: {bcop.weight.shape} vs {(output_channels, input_channels // groups, kernel_size, kernel_size)}"
        )
    # check that the layer is norm preserving
    inp_norm = torch.sqrt(torch.square(inp).sum(dim=(-3, -2, -1))).float().item()
    out_norm = torch.sqrt(torch.square(output).sum(dim=(-3, -2, -1))).float().item()
    if inp_norm <= out_norm - 1e-3:
        pytest.fail(
            f"BCOP is not norm preserving: {inp_norm} vs {out_norm} with rel error {abs(inp_norm - out_norm) / inp_norm}"
        )

    # Test singular_values function
    sigma_min, sigma_max, stable_rank = bcop.singular_values()  # try:
    sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
        bcop, (input_channels, imsize, imsize)
    )
    print(
        f"({input_channels}->{output_channels}, g{groups}, k{kernel_size}), "
        f"sigma_max:"
        f" {sigma_max:.3f}/{sigma_max_ir:.3f}, "
        f"sigma_min:"
        f" {sigma_min:.3f}/{sigma_min_ir:.3f}, "
        f"stable_rank: {stable_rank:.3f}/{stable_rank_ir:.3f}"
    )
    tol = 1e-4
    # check that the singular values are close to 1
    assert sigma_max_ir < (1 + tol), "sigma_max is not less than 1"
    assert (sigma_min_ir < (1 + tol)) and (
        sigma_min_ir > 0.95
    ), "sigma_min is not close to 1"
    assert abs(stable_rank_ir - 1) < tol, "stable_rank is not close to 1"
    # check that the singular values are close to the impulse response values
    assert (
        abs(sigma_max - sigma_max_ir) < tol
    ), f"sigma_max is not close to its IR value: {sigma_max} vs {sigma_max_ir}"
    assert (
        abs(sigma_min - sigma_min_ir) < tol
    ), f"sigma_min is not close to its IR value: {sigma_min} vs {sigma_min_ir}"
    assert (
        abs(stable_rank - stable_rank_ir) < tol
    ), f"stable_rank is not close to its IR value: {stable_rank} vs {stable_rank_ir}"
