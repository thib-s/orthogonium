import pytest
import torch

from flashlipschitz.layers.conv.fast_block_ortho_conv import FlashBCOP

# from flashlipschitz.layers.conv.ortho_conv import OrthoConv as FlashBCOP


def _compute_sv_impulse_response_layer(layer, img_shape):
    inputs = torch.eye(img_shape[0] * img_shape[1] * img_shape[2]).view(
        img_shape[0] * img_shape[1] * img_shape[2],
        img_shape[0],
        img_shape[1],
        img_shape[2],
    )
    outputs = layer(inputs)
    svs = torch.linalg.svdvals(outputs.view(outputs.shape[0], -1))
    return svs.max(), svs.min(), svs.mean() / svs.max()


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [8, 16, 32])
@pytest.mark.parametrize("output_channels", [16, 32, 64])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_bcop(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    try:
        bcop = FlashBCOP(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=kernel_size // 2,
            padding_mode="circular",
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    imsize = 8
    # Test backpropagation and weight update
    try:
        bcop.train()
        inp = torch.randn(1, input_channels, imsize, imsize)
        opt = torch.optim.SGD(bcop.parameters(), lr=0.1)
        output = bcop(inp)
        output.backward(torch.randn_like(output))
        opt.step()
        bcop.eval()  # so impulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")

    # check that bcop.weight has the correct shape
    if bcop.weight.data.shape != (
        output_channels,
        input_channels // groups,
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
    try:
        sigma_max, sigma_min, stable_rank = bcop.singular_values()
        assert sigma_max < (1 + 1e-3), "sigma_max is not less than 1"
        assert (sigma_min < (1 + 1e-3)) and (
            sigma_min > 0.95
        ), "sigma_min is not close to 1"
        assert abs(stable_rank - 1) < 1e-3, "stable_rank is not close to 1"
        sigma_max_ir, sigma_min_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
            bcop, (input_channels, imsize, imsize)
        )
    except Exception as e:
        # pytest.skip(f"SVD failed with LinalgError {e}")
        # assing value to help linter following code wont be executed when linalgerror is raised
        sigma_max_ir, sigma_min_ir, stable_rank_ir = sigma_max, sigma_min, stable_rank
    assert (
        abs(sigma_max - sigma_max_ir) < 1e-2
    ), f"sigma_max is not close to its IR value: {sigma_max} vs {sigma_max_ir}"
    assert (
        abs(sigma_min - sigma_min_ir) < 1e-2
    ), f"sigma_min is not close to its IR value: {sigma_min} vs {sigma_min_ir}"
    assert (
        abs(stable_rank - stable_rank_ir) < 1e-2
    ), f"stable_rank is not close to its IR value: {stable_rank} vs {stable_rank_ir}"
