import numpy as np
import pytest
import torch

from orthogonium.layers.conv.AOC.rko_conv import RKOConv2d
from orthogonium.layers.linear.reparametrizers import DEFAULT_TEST_ORTHO_PARAMS

# from orthogonium.layers.conv.fast_block_ortho_conv import FlashBCOP


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


def check_orthogonal_layer(
    orthoconv,
    groups,
    input_channels,
    kernel_size,
    output_channels,
    expected_kernel_shape,
    check_orthogonality=True,
):
    imsize = 8
    # Test backpropagation and weight update
    try:
        orthoconv.train()
        opt = torch.optim.SGD(orthoconv.parameters(), lr=0.001)
        for i in range(25):
            opt.zero_grad()
            inp = torch.randn(1, input_channels, imsize, imsize)
            output = orthoconv(inp)
            loss = -output.mean()
            loss.backward()
            opt.step()
        orthoconv.eval()  # so i    mpulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")
    # check that orthoconv.weight has the correct shape
    if orthoconv.weight.data.shape != expected_kernel_shape:
        pytest.fail(
            f"RKO weight has incorrect shape: {orthoconv.weight.shape} vs {(output_channels, input_channels // groups, kernel_size, kernel_size)}"
        )
    # check that the layer is norm preserving
    inp_norm = torch.sqrt(torch.square(inp).sum(dim=(-3, -2, -1))).float().item()
    out_norm = torch.sqrt(torch.square(output).sum(dim=(-3, -2, -1))).float().item()
    if check_orthogonality:
        if inp_norm <= out_norm - 1e-3:
            pytest.fail(
                f"RKO is not norm preserving: {inp_norm} vs {out_norm} with rel error {abs(inp_norm - out_norm) / inp_norm}"
            )
    # Test singular_values function
    sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
        orthoconv, (input_channels, imsize, imsize)
    )
    try:
        sigma_min, sigma_max, stable_rank = orthoconv.singular_values()
    except RuntimeError as e:
        if e.args[0].startswith(
            "Not able to compute singular values for this configuration"
        ):
            sigma_min, sigma_max, stable_rank = (
                sigma_min_ir,
                sigma_max_ir,
                stable_rank_ir,
            )
        else:
            pytest.fail(f"Error in singular_values method: {e}")
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
    if check_orthogonality:
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


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_standard_configs(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = RKOConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
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
        check_orthogonality=(kernel_size == stride),
    )


@pytest.mark.parametrize("kernel_size", [3, 4, 5])
@pytest.mark.parametrize("input_channels", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("output_channels", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("stride", [2, 4])
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
        orthoconv = RKOConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
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
        check_orthogonality=(kernel_size == stride),
    )


@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_even_kernels(kernel_size, input_channels, output_channels, stride, groups):
    """
    test specific to even kernel size
    """
    # Test instantiation
    try:
        orthoconv = RKOConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding="same",
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
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
        check_orthogonality=(kernel_size == stride),
    )


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("input_channels", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("output_channels", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("groups", [1, 2])
def test_rko(kernel_size, input_channels, output_channels, groups):
    """
    test case where stride == kernel size
    """
    # Test instantiation
    try:
        rkoconv = RKOConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=kernel_size,
            groups=groups,
            bias=False,
            padding=(0, 0),
            padding_mode="zeros",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        rkoconv,
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
        check_orthogonality=(kernel_size == kernel_size),
    )


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize("output_channels", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1])
def test_depthwise(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = RKOConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if input_channels == 1 and output_channels == 1:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
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
        check_orthogonality=(kernel_size == stride),
    )
