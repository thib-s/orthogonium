import pytest
import torch
import torch.nn.functional as F
import numpy as np
from flashlipschitz.layers.fast_block_ortho_conv import fast_matrix_conv

THRESHOLD = 1e-5


@pytest.mark.parametrize(
    "img_shape,kernel1_shape,kernel2_shape,stride,padding",
    [
        # Add tuples of the form (img_shape, kernel1_shape, kernel2_shape, stride, padding)
        # to test various configurations. For example:
        ((128, 32, 32, 3), (16, 16, 5, 3), (16, 16, 3, 5), (1, 1), "circular"),
        ((128, 32, 32, 3), (16, 32, 3, 3), (32, 64, 5, 5), (1, 1), "circular"),
        ((128, 32, 32, 3), (16, 16, 3, 3), (16, 16, 5, 5), (1, 1), "valid"),
        ((128, 32, 32, 3), (16, 32, 5, 3), (32, 64, 3, 5), (1, 1), "valid"),
        ((128, 32, 32, 3), (16, 16, 3, 3), (16, 16, 5, 5), (1, 2), "valid"),
        ((128, 32, 32, 3), (16, 32, 5, 5), (32, 64, 3, 3), (1, 2), "circular"),
        # Add more configurations as needed
    ],
)
def test_conv2d_operations(
    img_shape, kernel1_shape, kernel2_shape, stride_1, stride_2, padding
):
    # Initialize your kernels
    kernel_1 = torch.randn(kernel1_shape)
    kernel_2 = torch.randn(kernel2_shape)
    kernel_merged = fast_matrix_conv(kernel_1, kernel_2)

    # Initialize a dummy image based on img_shape
    img = torch.tensor(np.random.random(img_shape).astype(np.float32))

    # Perform your convolution operations as in the original code
    if padding not in [None, "same", "valid"]:
        img_padded = F.pad(img, padding)
        res_1 = F.conv2d(img_padded, kernel_1, stride=1, padding="valid")
    else:
        res_1 = F.conv2d(img, kernel_1, stride=1, padding=padding)
    if padding not in [None, "same", "valid"]:
        res_1_padded = F.pad(res_1, padding)
        res_1 = F.conv2d(res_1_padded, kernel_2, stride=1, padding="valid")
    else:
        res_1 = F.conv2d(res_1, kernel_2, stride=stride_2, padding=padding)

    # Your convolution and comparison logic here
    res_2 = F.conv2d(torch.tensor(img), kernel_merged, stride=stride_2, padding=padding)
    # Use an assert statement to check if the result meets your expectation
    # For example, checking if the difference between res_1 and res_2 is below a threshold
    assert torch.mean(torch.square(res_1 - res_2)) < THRESHOLD
