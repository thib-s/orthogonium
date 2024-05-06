import pytest
import torch
import numpy as np

from flashlipschitz.layers import OrthoConv2d

# from flashlipschitz.layers.conv.fast_block_ortho_conv import FlashBCOP
from flashlipschitz.layers.conv.reparametrizers import BjorckParams


def check_expressiveness_layer(
    orthoconv,
    input_channels,
    imsize,
    target_weight,
):
    # Test backpropagation and weight update
    try:
        orthoconv.train()
        opt = torch.optim.Adam(orthoconv.parameters(), lr=0.05)
        for i in range(250):
            opt.zero_grad()
            inp = torch.randn(1, input_channels, imsize, imsize)
            output = orthoconv(inp)
            loss = torch.norm(orthoconv.weight - target_weight, p="fro")
            loss.backward()
            opt.step()
        orthoconv.eval()  # so impulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")
    print(f"max diff: {torch.max(torch.abs(orthoconv.weight - target_weight)):0.4f}")
    assert torch.allclose(
        orthoconv.weight, target_weight, atol=0.05
    ), (f"layer failed to converge to target weight. Max diff: "
        f"{torch.max(torch.abs(orthoconv.weight - target_weight)):0.4f}")


@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("input_channels", [1, 4])
@pytest.mark.parametrize("output_channels", [1, 4])
@pytest.mark.parametrize("shift_x", [-1, 0, 1])
@pytest.mark.parametrize("shift_y", [-1, 0, 1])
def test_expressiveness_shifts(
    kernel_size, input_channels, output_channels, shift_x, shift_y
):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    orthoconv = OrthoConv2d(
        kernel_size=kernel_size,
        in_channels=input_channels,
        out_channels=output_channels,
        stride=1,
        groups=1,
        bias=False,
        padding=(kernel_size // 2, kernel_size // 2),
        padding_mode="circular",
        bjorck_params=BjorckParams(
            power_it_niter=3,
            eps=1e-6,
            beta=0.5,
            bjorck_iters=20,
        ),
    )
    target_weight = torch.eye(output_channels, input_channels).view(
        output_channels, input_channels, 1, 1
    )
    target_weight = torch.nn.functional.pad(
        target_weight,
        (
            (kernel_size - 1) // 2,
            (kernel_size - 1) // 2,
            (kernel_size - 1) // 2,
            (kernel_size - 1) // 2,
        ),
    )
    target_weight = torch.roll(target_weight, shifts=(shift_x, shift_y), dims=(2, 3))
    check_expressiveness_layer(
        orthoconv,
        input_channels,
        imsize=4,
        target_weight=target_weight,
    )