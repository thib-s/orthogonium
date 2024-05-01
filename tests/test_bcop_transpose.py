import numpy as np
import pytest
import torch

from flashlipschitz.layers.conv.bcop_x_rko_conv import BcopRkoConvTranspose2d
from flashlipschitz.layers.conv.ortho_conv import OrthoConvTranspose2d
from flashlipschitz.layers.conv.reparametrizers import BjorckParams
from tests.test_bcop import check_orthogonal_layer


# from flashlipschitz.layers.conv.fast_block_ortho_conv import FlashBCOP

# from flashlipschitz.layers.conv.ortho_conv import OrthoConv as FlashBCOP


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


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("input_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("output_channels", [4, 8, 16, 32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0)
    padding_mode = "zeros"
    try:
        if (
            kernel_size > 1
            and kernel_size != stride
            and output_channels * (stride**2) < input_channels
        ):
            with pytest.warns(RuntimeWarning):
                orthoconvtranspose = OrthoConvTranspose2d(
                    kernel_size=kernel_size,
                    in_channels=input_channels,
                    out_channels=output_channels,
                    stride=stride,
                    groups=groups,
                    bias=False,
                    padding=padding,
                    padding_mode=padding_mode,
                    bjorck_params=BjorckParams(
                        power_it_niter=10,
                        eps=1e-6,
                        beta=0.5,
                        bjorck_iters=50,
                    ),
                )
        else:
            orthoconvtranspose = OrthoConvTranspose2d(
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                stride=stride,
                groups=groups,
                bias=False,
                padding=padding,
                padding_mode=padding_mode,
                bjorck_params=BjorckParams(
                    power_it_niter=10,
                    eps=1e-6,
                    beta=0.5,
                    bjorck_iters=50,
                ),
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
        pytest.skip("this cas is not handled yet")
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
    )
