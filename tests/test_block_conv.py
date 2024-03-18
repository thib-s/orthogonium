import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flashlipschitz.layers.fast_block_ortho_conv import fast_matrix_conv

THRESHOLD = 1e-5


# note that only square kernels are tested here
# implementation seems to be correct for rectangular kernels as well
# but the test cases fail with padding='circular' du to poor padding
# implementation withing this test
@pytest.mark.parametrize("kernel_size_1", [3, 5])
@pytest.mark.parametrize("kernel_size_2", [3, 5])
@pytest.mark.parametrize("channels_1", [8, 16])
@pytest.mark.parametrize("channels_2", [8, 16])
@pytest.mark.parametrize("channels_3", [8, 16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", ["valid", "circular"])
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
    # Initialize your kernels
    img_shape = (128, channels_1, 32, 32)
    kernel1_shape = (channels_2, channels_1 // groups, kernel_size_1, kernel_size_1)
    kernel2_shape = (channels_3, channels_2 // groups, kernel_size_2, kernel_size_2)
    kernel_1 = torch.randn(kernel1_shape)
    kernel_2 = torch.randn(kernel2_shape)
    kernel_merged = fast_matrix_conv(kernel_1, kernel_2, groups=groups)

    # Initialize a dummy image based on img_shape
    img = torch.randn(img_shape)

    # Perform your convolution operations as in the original code
    if padding not in [None, "same", "valid"]:
        img_padded = F.pad(
            img,
            (
                kernel_1.shape[-2] // 2,
                kernel_1.shape[-1] // 2,
            )
            * 2,
            mode=padding,
        )
        res_1 = F.conv2d(img_padded, kernel_1, stride=1, groups=groups, padding="valid")
    else:
        res_1 = F.conv2d(img, kernel_1, stride=1, groups=groups, padding=padding)
    if padding not in [None, "same", "valid"]:
        res_1_padded = F.pad(
            res_1,
            (
                kernel_2.shape[-2] // 2,
                kernel_2.shape[-1] // 2,
            )
            * 2,
            mode=padding,
        )
        res_1 = F.conv2d(
            res_1_padded, kernel_2, stride=1, groups=groups, padding="valid"
        )
    else:
        res_1 = F.conv2d(res_1, kernel_2, stride=stride, groups=groups, padding=padding)

    if padding not in [None, "same", "valid"]:
        img_padded = F.pad(
            img,
            (
                kernel_merged.shape[-2] // 2,
                kernel_merged.shape[-1] // 2,
            )
            * 2,
            mode=padding,
        )
        res_2 = F.conv2d(
            img_padded, kernel_merged, stride=1, groups=groups, padding="valid"
        )
    else:
        res_2 = F.conv2d(
            img, kernel_merged, stride=stride, groups=groups, padding=padding
        )
    # Use an assert statement to check if the result meets your expectation
    # For example, checking if the difference between res_1 and res_2 is below a threshold
    assert torch.mean(torch.square(res_1 - res_2)) < THRESHOLD
