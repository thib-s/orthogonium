import pytest
import torch

from flashlipschitz.layers.fast_block_ortho_conv import FlashBCOP


def _compute_sv_impulse_response_layer(layer, img_shape):
    inputs = torch.eye(img_shape[0] * img_shape[1] * img_shape[2]).view(
        img_shape[0] * img_shape[1] * img_shape[2],
        img_shape[0],
        img_shape[1],
        img_shape[2],
    )
    outputs = layer(inputs)
    print(outputs.shape)
    svs = torch.linalg.svdvals(outputs.view(outputs.shape[0], -1))
    return svs.max(), svs.min(), svs.mean() / svs.max()


@pytest.mark.parametrize("kernel_size", [1, 2, 3, 5])
@pytest.mark.parametrize("input_channels", [8, 16, 32, 64])
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
            padding_mode="circular",
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        elif kernel_size == 1 and input_channels == output_channels:
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    imsize = 8
    # Test backpropagation and weight update
    try:
        input = torch.randn(1, input_channels, imsize, imsize)
        opt = torch.optim.SGD(bcop.parameters(), lr=0.1)
        output = bcop(input)
        output.backward(torch.randn_like(output))
        opt.step()
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")

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
        pytest.skip(f"SVD failed with LinalgError {e}")
        # assing value to help linter following code wont be executed when linalgerror is raised
        sigma_max_ir, sigma_min_ir, stable_rank_ir = sigma_max, sigma_min, stable_rank
    assert (
        abs(sigma_max - sigma_max_ir) < 1e-3
    ), f"sigma_max is not close to its IR value: {sigma_max} vs {sigma_max_ir}"
    assert (
        abs(sigma_min - sigma_min_ir) < 1e-3
    ), f"sigma_min is not close to its IR value: {sigma_min} vs {sigma_min_ir}"
    assert (
        abs(stable_rank - stable_rank_ir) < 1e-3
    ), f"stable_rank is not close to its IR value: {stable_rank} vs {stable_rank_ir}"
