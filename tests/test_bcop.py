import pytest
import torch

from flashlipschitz.layers.fast_block_ortho_conv import BCOP


@pytest.mark.parametrize("kernel_size", [1, 3, 5, 7])
@pytest.mark.parametrize("input_channels", [8, 16, 32])
@pytest.mark.parametrize("output_channels", [8, 16, 32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_bcop(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    try:
        bcop = BCOP(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        elif kernel_size == 1 and input_channels == output_channels:
            pass
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")

    # Test backpropagation and weight update
    try:
        input = torch.randn(1, input_channels, 32, 32)
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
    except Exception as e:
        pytest.fail(f"singular_values function failed with: {e}")
