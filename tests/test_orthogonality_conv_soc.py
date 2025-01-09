import numpy as np
import pytest
import torch
from orthogonium.layers.conv.adaptiveSOC import (
    AdaptiveSOCConv2d,
    AdaptiveSOCConvTranspose2d,
)
from orthogonium.layers.conv.adaptiveSOC.soc_x_rko_conv import SOCRkoConv2d
from orthogonium.layers.conv.adaptiveSOC.fast_skew_ortho_conv import FastSOC


device = "cpu"  #  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_sv_impulse_response_layer(layer, img_shape):
    with torch.no_grad():
        layer = layer.to(device)
        inputs = (
            torch.eye(img_shape[0] * img_shape[1] * img_shape[2])
            .view(
                img_shape[0] * img_shape[1] * img_shape[2],
                img_shape[0],
                img_shape[1],
                img_shape[2],
            )
            .to(device)
        )
        outputs = layer(inputs)
        try:
            svs = torch.linalg.svdvals(outputs.view(outputs.shape[0], -1))
            svs = svs.cpu()
            return svs.min(), svs.max(), svs.mean() / svs.max()
        except np.linalg.LinAlgError:
            print("SVD failed returning only largest singular value")
            return torch.norm(outputs.view(outputs.shape[0], -1), p=2).max(), 0, 0


def check_orthogonal_layer(
    orthoconv,
    groups,
    input_channels,
    kernel_size,
    output_channels,
    expected_kernel_shape,
    tol=5e-4,
    sigma_min_requirement=0.95,
):
    imsize = 8
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
    # # check that orthoconv.weight has the correct shape
    # if orthoconv.weight.data.shape != expected_kernel_shape:
    #     pytest.fail(
    #         f"BCOP weight has incorrect shape: {orthoconv.weight.shape} vs {(output_channels, input_channels // groups, kernel_size, kernel_size)}"
    #     )
    # Test singular_values function
    try:
        sigma_min, sigma_max, stable_rank = orthoconv.singular_values()  # try:
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
        f" {sigma_min:.3f}/{sigma_min_ir:.3f}, "
        f"stable_rank: {stable_rank:.3f}/{stable_rank_ir:.3f}"
    )
    # check that the singular values are close to 1
    assert sigma_max_ir < (1 + tol), "sigma_max is not less than 1"
    # assert (sigma_min_ir < (1 + tol)) and (
    #     sigma_min_ir > sigma_min_requirement
    # ), "sigma_min is not close to 1"
    # assert abs(stable_rank_ir - 1) < tol, "stable_rank is not close to 1"
    # check that the singular values are close to the impulse response values
    # assert (
    #     sigma_max > sigma_max_ir - 1e-2
    # ), f"sigma_max must be greater to its IR value (1%): {sigma_max} vs {sigma_max_ir}"
    assert (
        abs(sigma_max - sigma_max_ir) < tol
    ), f"sigma_max is not close to its IR value: {sigma_max} vs {sigma_max_ir}"
    # assert (
    #     abs(sigma_min - sigma_min_ir) < tol
    # ), f"sigma_min is not close to its IR value: {sigma_min} vs {sigma_min_ir}"
    # assert (
    #     abs(stable_rank - stable_rank_ir) < tol
    # ), f"stable_rank is not close to its IR value: {stable_rank} vs {stable_rank_ir}"


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
            padding=(kernel_size // 2, kernel_size // 2),
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
        tol=5e-2,
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
            padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
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
        tol=5e-2,
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
            padding=(kernel_size // 2, kernel_size // 2),
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
        tol=5e-2,
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
def test_convtranspose(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0)
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
        tol=5e-2,
        sigma_min_requirement=0.0,
    )


@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("input_channels", [4, 8])
@pytest.mark.parametrize("output_channels", [4, 8])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0)
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
        tol=5e-2,
        sigma_min_requirement=0.0,
    )
