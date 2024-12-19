import pytest
import torch
from orthogonium.layers.channel_shuffle import ChannelShuffle


def test_forward_output_shapes():
    # Test case 1: 2 groups of size 3 -> 3 groups of size 2
    x = torch.randn(2, 6)
    gm = ChannelShuffle(2, 3)
    y = gm(x)

    assert y.size() == torch.Size([2, 6])  # Output should have the same shape as input

    # Test case 2: 2 groups of size 3 (4D tensor) -> 3 groups of size 2
    x2 = torch.randn(2, 6, 32, 32)
    gm = ChannelShuffle(2, 3)
    y2 = gm(x2)

    assert y2.shape == (2, 6, 32, 32)  # Check that the shape remains unchanged


def test_channel_shuffle_invertibility():
    # Test that ChannelShuffle is invertible
    x2 = torch.randn(2, 6, 32, 32)
    gm = ChannelShuffle(2, 3)
    y2 = gm(x2)
    gp = ChannelShuffle(3, 2)  # Invert the shuffle step
    x2b = gp(y2)

    assert torch.allclose(x2, x2b), "ChannelShuffle is not invertible"


def test_invalid_tensor_size():
    # Test case where input tensor size doesn't match group_in * group_out
    x_invalid = torch.randn(2, 5)  # 5 does not match 2 * 3
    gm = ChannelShuffle(2, 3)
    with pytest.raises(AssertionError):
        gm(x_invalid)


def test_invalid_dim():
    # Test case where dim is not equal to 1
    with pytest.raises(AssertionError):
        ChannelShuffle(2, 3, dim=2)


def test_extra_repr():
    # Test the extra_repr output
    gm = ChannelShuffle(2, 3)
    assert gm.extra_repr() == "group_in=2, group_out=3"


def test_channel_shuffle_1_lipschitz():
    # Initialize the ChannelShuffle layer
    group_in, group_out = 2, 3
    gm = ChannelShuffle(group_in, group_out)

    # Input tensor (requires gradient for Jacobian computation)
    x = torch.randn(2, group_in * group_out, requires_grad=True)

    # Forward pass
    y = gm(x)

    # Compute the Jacobian matrix using autograd
    jacobian = []
    for i in range(y.numel()):  # Iterate over each output element
        grad_output = torch.zeros_like(y)
        grad_output.view(-1)[i] = 1  # Set gradient w.r.t. one output element
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        jacobian.append(gradients.view(-1).detach().cpu().numpy())
    jacobian = torch.tensor(jacobian).view(y.numel(), x.numel())  # Assemble Jacobian

    # Compute the spectral norm of the Jacobian
    singular_values = torch.linalg.svdvals(jacobian)
    spectral_norm = singular_values.max()
    min_singular_value = singular_values.min().item()

    # Check that the spectral norm is <= 1
    assert spectral_norm <= 1 + 1e-4, "ChannelShuffle is not 1-Lipschitz"
    assert (
        pytest.approx(min_singular_value, rel=1e-6) == 1
    ), "ChannelShuffle is not orthogonal"
