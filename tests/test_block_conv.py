import pytest
import torch

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_batched_matrix_conv
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_matrix_conv

THRESHOLD = 5e-4


# note that only square kernels are tested here
# implementation seems to be correct for rectangular kernels as well
# but the test cases fail with padding='circular' du to poor padding
# implementation withing this test
@pytest.mark.parametrize("kernel_size_1", [3, 5])
@pytest.mark.parametrize("kernel_size_2", [3, 5])
@pytest.mark.parametrize("channels_1", [4, 8])
@pytest.mark.parametrize("channels_2", [4, 8])
@pytest.mark.parametrize("channels_3", [4, 8])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["valid-circular", "same-circular"])
@pytest.mark.parametrize("groups", [1, 4])
def test_conv2d_operations(
    kernel_size_1,
    kernel_size_2,
    channels_1,
    channels_2,
    channels_3,
    stride,
    padding,
    groups,
):
    padding, padding_mode = padding.split("-")
    padding = 0 if stride != 1 else padding
    # Initialize your kernels
    img_shape = (128, channels_1, 32, 32)
    kernel1_shape = (channels_2, channels_1 // groups, kernel_size_1, kernel_size_1)
    kernel2_shape = (channels_3, channels_2 // groups, kernel_size_2, kernel_size_2)
    kernel_1 = torch.randn(kernel1_shape)
    kernel_2 = torch.randn(kernel2_shape)
    conv_1 = torch.nn.Conv2d(
        channels_1,
        channels_2,
        kernel_size_1,
        stride=1,
        padding=padding,
        padding_mode=padding_mode,
        groups=groups,
        bias=False,
    )
    conv_1.weight.data = kernel_1
    conv_2 = torch.nn.Conv2d(
        channels_2,
        channels_3,
        kernel_size_2,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=groups,
        bias=False,
    )
    conv_2.weight.data = kernel_2

    kernel_merged = fast_matrix_conv(kernel_1, kernel_2, groups=groups)
    conv_merged = torch.nn.Conv2d(
        channels_1,
        channels_3,
        kernel_merged.shape[-1],
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=groups,
        bias=False,
    )
    conv_merged.weight.data = kernel_merged
    # Initialize a dummy image based on img_shape
    img = torch.randn(img_shape)

    # Perform your convolution operations as in the original code
    res_1 = conv_1(img)
    res_1 = conv_2(res_1)
    res_2 = conv_merged(img)
    assert torch.mean(torch.square(res_1 - res_2)) < THRESHOLD


@pytest.mark.parametrize("kernel_size_1", [3, 5])
@pytest.mark.parametrize("kernel_size_2", [3, 5])
@pytest.mark.parametrize("channels_1", [8, 16])
@pytest.mark.parametrize("channels_2", [8, 16])
@pytest.mark.parametrize("channels_3", [8, 16, 32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["valid-circular", "same-circular"])
@pytest.mark.parametrize("groups", [1, 4])
def test_batched_conv2d_operations(
    kernel_size_1,
    kernel_size_2,
    channels_1,
    channels_2,
    channels_3,
    stride,
    padding,
    groups,
):
    batch_size = 4
    padding, padding_mode = padding.split("-")
    padding = 0 if stride != 1 else padding
    # Initialize your kernels
    kernel1_shape = (
        batch_size,
        channels_2,
        channels_1 // groups,
        kernel_size_1,
        kernel_size_1,
    )
    kernel2_shape = (
        batch_size,
        channels_3,
        channels_2 // groups,
        kernel_size_2,
        kernel_size_2,
    )
    kernel_1 = torch.randn(kernel1_shape)
    kernel_2 = torch.randn(kernel2_shape)
    res1 = torch.stack(
        [
            fast_matrix_conv(kernel_1[i], kernel_2[i], groups=groups)
            for i in (range(batch_size))
        ],
        dim=0,
    )
    res2 = fast_batched_matrix_conv(kernel_1, kernel_2, groups=groups)
    assert torch.mean(torch.square(res1 - res2)) < THRESHOLD
