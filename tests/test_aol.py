import pytest
import torch
from orthogonium.layers.conv.AOL.aol import AOLConv2D, AOLConvTranspose2D
from orthogonium.layers.conv.singular_values import get_conv_sv


@pytest.mark.parametrize("convclass", [AOLConv2D, AOLConvTranspose2D])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("groups", [1, 4])
def test_lipschitz_layers(convclass, in_channels, out_channels, kernel_size, groups):
    """
    Generalized test for layers in the AOL module to check Lipschitz constraints.
    """
    # Initialize layer
    layer = convclass(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
    )

    # Define input and target tensors
    x = torch.randn((4, in_channels, 8, 8), requires_grad=True)  # Input

    # Pre-optimization Lipschitz constant (if applicable)
    pre_lipschitz_constant = get_conv_sv(layer, n_iter=5, agg_groups=True)
    print(f"{convclass.__name__} | Before: {pre_lipschitz_constant}")
    assert (
        pre_lipschitz_constant[0] <= 1 + 1e-4
    ), "Pre-optimization Lipschitz constant violation."

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    # Perform a few optimization steps
    for _ in range(10):  # Run 10 optimization steps
        optimizer.zero_grad()
        output = layer(x)
        loss = -torch.sum(torch.square(output))
        loss.backward()
        optimizer.step()

    # Post-optimization Lipschitz constant (if applicable)
    post_lipschitz_constant = get_conv_sv(layer, n_iter=5, agg_groups=True)
    print(f"{convclass.__name__} | After: {post_lipschitz_constant}")
    assert (
        post_lipschitz_constant[0] <= 1 + 1e-4
    ), "Post-optimization Lipschitz constant violation."
