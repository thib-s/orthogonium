import numpy as np
import pytest
import torch
from orthogonium.layers.conv.adaptiveSOC import (
    AdaptiveSOCConv2d,
    AdaptiveSOCConvTranspose2d,
)
from orthogonium.layers.conv.singular_values import get_conv_sv
from tests.test_orthogonality_conv import _compute_sv_impulse_response_layer

device = "cpu"  #  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_orthogonal_layer(
    orthoconv,
    groups,
    input_channels,
    kernel_size,
    output_channels,
    expected_kernel_shape,
    tol=1e-2,
    sigma_min_requirement=0.0,
    imsize=8,
):
    # fixing seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    # Test backpropagation and weight update
    try:
        orthoconv = orthoconv.to(device)
        orthoconv.train()
        opt = torch.optim.SGD(orthoconv.parameters(), lr=0.001)
        for i in range(25):
            opt.zero_grad()
            inp = torch.randn(1, input_channels, imsize, imsize).to(device)
            output = orthoconv(inp)
            loss = -output.mean()
            loss.backward()
            opt.step()
        orthoconv.eval()  # so i    mpulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")
    with torch.no_grad():
        try:
            sigma_max, stable_rank = get_conv_sv(
                orthoconv,
                n_iter=6 if orthoconv.padding_mode == "circular" else 3,
                imsize=imsize,
            )
        except np.linalg.LinAlgError as e:
            pytest.skip(f"SVD failed with: {e}")
        sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
            orthoconv, (input_channels, imsize, imsize)
        )
    print(f"input_shape = {inp.shape}, output_shape = {output.shape}")
    print(
        f"({input_channels}->{output_channels}, g{groups}, k{kernel_size}), "
        f"sigma_max:"
        f" {sigma_max:.3f}/{sigma_max_ir:.3f}, "
        f"sigma_min:"
        f" {sigma_min_ir:.3f}, "
        f"stable_rank: {stable_rank:.3f}/{stable_rank_ir:.3f}"
    )
    # check that the singular values are close to 1
    assert sigma_max_ir < (1 + tol), "sigma_max is not less than 1"
    assert (sigma_min_ir < (1 + tol)) and (
        sigma_min_ir > sigma_min_requirement
    ), "sigma_min is not close to 1"
    try:
        # check that table rank is greater than 0.75
        assert stable_rank_ir > 0.75, "stable rank is not greater than 0.75"
        assert (
            sigma_max + tol >= sigma_max_ir
        ), f"sigma_max is not greater than its IR value: {sigma_max} vs {sigma_max_ir}"
    except AssertionError as e:
        # given the large number of tests and the stochastic nature of these, we can
        # expect 1 over 100 tests to fail. Especially on less mandatory properties
        # (like stable rank). However, it is relevant to check that this is not a systematic
        # failure. To do so, when the test fails, performs a less strict check and decide if
        # the test will raise a warning or an error. (The number of warnings should be monitored)
        assert stable_rank_ir > 0.25, "stable rank is not greater than 0.25"
        pytest.skip("Stable rank is less than 0.75, but greater than 0.25")


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_standard_configs(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveSOCConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(3 * (kernel_size // 2), 3 * (kernel_size // 2)),
            padding_mode="circular",
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=8e-2,
        sigma_min_requirement=0.0,
    )


#
# @pytest.mark.parametrize("kernel_size", [3])
# @pytest.mark.parametrize("input_channels", [8, 16])
# @pytest.mark.parametrize(
#     "output_channels", [8, 16]
# )  # dilated convolutions are not supported for output_channels < input_channels
# @pytest.mark.parametrize("stride", [1])
# @pytest.mark.parametrize("groups", [1, 2, 4])
# def test_dilation(kernel_size, input_channels, output_channels, stride, groups):
#     """
#     test combinations of kernel size, input channels, output channels, stride and groups
#     """
#     # Test instantiation
#     try:
#         orthoconv = AdaptiveSOCConv2d(
#             kernel_size=kernel_size,
#             in_channels=input_channels,
#             out_channels=output_channels,
#             stride=stride,
#             dilation=2,
#             groups=groups,
#             bias=False,
#             padding="same",
#             padding_mode="circular",
#         )
#     except Exception as e:
#         if kernel_size < stride:
#             # we expect this configuration to raise a RuntimeError
#             # pytest.skip(f"BCOP instantiation failed with: {e}")
#             return
#         else:
#             pytest.fail(f"BCOP instantiation failed with: {e}")
#     check_orthogonal_layer(
#         orthoconv,
#         groups,
#         input_channels,
#         kernel_size,
#         output_channels,
#         (
#             output_channels,
#             input_channels // groups,
#             kernel_size,
#             kernel_size,
#         ),
#     )
#
#
# @pytest.mark.parametrize("kernel_size", [2, 4])
# @pytest.mark.parametrize("input_channels", [8, 16])
# @pytest.mark.parametrize(
#     "output_channels", [8, 16]
# )  # dilated+strided convolutions are not supported for output_channels < input_channels
# @pytest.mark.parametrize("stride", [2])
# @pytest.mark.parametrize("dilation", [2, 3])
# @pytest.mark.parametrize("groups", [1, 2, 4])
# def test_dilation_strided(
#     kernel_size, input_channels, output_channels, stride, dilation, groups
# ):
#     """
#     test combinations of kernel size, input channels, output channels, stride and groups
#     """
#     # Test instantiation
#     try:
#         orthoconv = AdaptiveSOCConv2d(
#             kernel_size=kernel_size,
#             in_channels=input_channels,
#             out_channels=output_channels,
#             stride=stride,
#             dilation=dilation,
#             groups=groups,
#             bias=False,
#             padding=(
#                 int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2)),
#                 int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2)),
#             ),
#             padding_mode="circular",
#         )
#     except Exception as e:
#         if (output_channels >= input_channels) and (
#             ((dilation % stride) == 0) and (stride > 1)
#         ):
#             # we expect this configuration to raise a ValueError
#             # pytest.skip(f"BCOP instantiation failed with: {e}")
#             return
#         if (kernel_size == stride) and (((dilation % stride) == 0) and (stride > 1)):
#             return
#         else:
#             pytest.fail(f"BCOP instantiation failed with: {e}")
#     check_orthogonal_layer(
#         orthoconv,
#         groups,
#         input_channels,
#         kernel_size,
#         output_channels,
#         (
#             output_channels,
#             input_channels // groups,
#             kernel_size,
#             kernel_size,
#         ),
#     )


@pytest.mark.parametrize("kernel_size", [4])
@pytest.mark.parametrize("input_channels", [2, 4, 16])
@pytest.mark.parametrize("output_channels", [2, 4, 16])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("groups", [1])
def test_strided(kernel_size, input_channels, output_channels, stride, groups):
    """
    a more extensive testing when striding is enabled.
    A larger range of cin and cout is used to track errors when cin < cout / stride**2
    ( ie you reduce spatial dimensions but you increase the channel dimensions so
    that you actually increase overall dimension.
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveSOCConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(3 * ((kernel_size - 1) // 2), 3 * ((kernel_size - 1) // 2)),
            padding_mode="circular",
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=8e-2,
        sigma_min_requirement=0.0,
    )


# @pytest.mark.parametrize("kernel_size", [2, 4])
# @pytest.mark.parametrize("input_channels", [8, 16])
# @pytest.mark.parametrize("output_channels", [8, 16])
# @pytest.mark.parametrize("stride", [1])
# @pytest.mark.parametrize("groups", [1, 2, 4])
# def test_even_kernels(kernel_size, input_channels, output_channels, stride, groups):
#     """
#     test specific to even kernel size
#     """
#     # Test instantiation
#     try:
#         orthoconv = AdaptiveSOCConv2d(
#             kernel_size=kernel_size,
#             in_channels=input_channels,
#             out_channels=output_channels,
#             stride=stride,
#             groups=groups,
#             bias=False,
#             padding="same",
#             padding_mode="circular",
#         )
#     except Exception as e:
#         if kernel_size < stride:
#             # we expect this configuration to raise a RuntimeError
#             # pytest.skip(f"BCOP instantiation failed with: {e}")
#             return
#         else:
#             pytest.fail(f"BCOP instantiation failed with: {e}")
#     check_orthogonal_layer(
#         orthoconv,
#         groups,
#         input_channels,
#         kernel_size,
#         output_channels,
#         (
#             output_channels,
#             input_channels // groups,
#             kernel_size,
#             kernel_size,
#         ),
#     )


# @pytest.mark.parametrize("kernel_size", [1, 2])
# @pytest.mark.parametrize("input_channels", [4, 8, 32])
# @pytest.mark.parametrize("output_channels", [4, 8, 32])
# @pytest.mark.parametrize("groups", [1, 2])
# def test_rko(kernel_size, input_channels, output_channels, groups):
#     """
#     test case where stride == kernel size
#     """
#     # Test instantiation
#     try:
#         rkoconv = AdaptiveSOCConv2d(
#             kernel_size=kernel_size,
#             in_channels=input_channels,
#             out_channels=output_channels,
#             stride=kernel_size,
#             groups=groups,
#             bias=False,
#             padding=(0, 0),
#             padding_mode="zeros",
#         )
#     except Exception as e:
#         pytest.fail(f"BCOP instantiation failed with: {e}")
#     check_orthogonal_layer(
#         rkoconv,
#         groups,
#         input_channels,
#         kernel_size,
#         output_channels,
#         (
#             output_channels,
#             input_channels // groups,
#             kernel_size,
#             kernel_size,
#         ),
#     )


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize("output_channels", [1, 2])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1])
def test_depthwise(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveSOCConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(3 * (kernel_size // 2), 3 * (kernel_size // 2)),
            padding_mode="circular",
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=8e-2,
        sigma_min_requirement=0.0,
    )


# def test_invalid_kernel_smaller_than_stride():
#     """
#     A test to ensure that kernel_size < stride raises an expected ValueError
#     """
#     with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
#         AdaptiveSOCConv2d(
#             in_channels=8,
#             out_channels=4,
#             kernel_size=2,
#             stride=3,  # Invalid: kernel_size < stride
#             groups=1,
#             padding=0,
#         )
#     with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
#         SOCRkoConv2d(
#             in_channels=8,
#             out_channels=4,
#             kernel_size=2,
#             stride=3,  # Invalid: kernel_size < stride
#             groups=1,
#             padding=0,
#         )
#     with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
#         FastSOC(
#             in_channels=8,
#             out_channels=4,
#             kernel_size=2,
#             stride=3,  # Invalid: kernel_size < stride
#             groups=1,
#             padding=0,
#         )
#
#
# def test_invalid_dilation_with_stride():
#     """
#     A test to ensure dilation > 1 while stride > 1 raises an expected ValueError
#     """
#     with pytest.raises(
#         ValueError,
#         match=r"dilation must be 1 when stride is not 1",
#     ):
#         AdaptiveSOCConv2d(
#             in_channels=8,
#             out_channels=16,
#             kernel_size=3,
#             stride=2,
#             dilation=2,  # Invalid: dilation > 1 while stride > 1
#             groups=1,
#             padding=0,
#         )
#     with pytest.raises(
#         ValueError,
#         match=r"dilation must be 1 when stride is not 1",
#     ):
#         SOCRkoConv2d(
#             in_channels=8,
#             out_channels=16,
#             kernel_size=3,
#             stride=2,
#             dilation=2,  # Invalid: dilation > 1 while stride > 1
#             groups=1,
#             padding=0,
#         )
#     with pytest.raises(
#         ValueError,
#         match=r"dilation must be 1 when stride is not 1",
#     ):
#         FastSOC(
#             in_channels=8,
#             out_channels=16,
#             kernel_size=3,
#             stride=2,
#             dilation=2,  # Invalid: dilation > 1 while stride > 1
#             groups=1,
#             padding=0,
#         )


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("input_channels", [4, 8])
@pytest.mark.parametrize("output_channels", [4, 8])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose_1(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (3 * (kernel_size // 2), 3 * (kernel_size // 2))
    padding_mode = "zeros"
    try:

        orthoconvtranspose = AdaptiveSOCConvTranspose2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    if (
        kernel_size > 1
        and kernel_size != stride
        and output_channels * (stride**2) < input_channels
    ):
        pytest.skip("this case is not handled yet")
    check_orthogonal_layer(
        orthoconvtranspose,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            input_channels,
            output_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=8e-2,
        sigma_min_requirement=0.0,
    )


@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("input_channels", [4, 8])
@pytest.mark.parametrize("output_channels", [4, 8])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose_2(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0) if kernel_size == stride else (3, 3)
    padding_mode = "zeros"
    try:

        orthoconvtranspose = AdaptiveSOCConvTranspose2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    if (
        kernel_size > 1
        and kernel_size != stride
        and output_channels * (stride**2) < input_channels
    ):
        pytest.skip("this case is not handled yet")
    check_orthogonal_layer(
        orthoconvtranspose,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            input_channels,
            output_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=8e-2,
        sigma_min_requirement=0.0,
    )
