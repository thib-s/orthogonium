import pytest
import torch
from orthogonium.layers.conv.AOL.aol import AOLConv2D, AOLConvTranspose2D


@pytest.mark.parametrize("convclass", [AOLConv2D, AOLConvTranspose2D])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("kernel_size", [2, 3, 5])
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
    x = torch.randn((8, in_channels, 8, 8), requires_grad=True)  # Input

    # Pre-optimization Lipschitz constant (if applicable)
    pre_lipschitz_constant = compute_lipschitz_constant(layer, x)
    print(f"{convclass.__name__} | Before: {pre_lipschitz_constant:.6f}")
    assert (
        pre_lipschitz_constant <= 1 + 1e-4
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
    post_lipschitz_constant = compute_lipschitz_constant(layer, x)
    print(f"{convclass.__name__} | After: {post_lipschitz_constant:.6f}")
    assert (
        post_lipschitz_constant <= 1 + 1e-4
    ), "Post-optimization Lipschitz constant violation."


def compute_lipschitz_constant(layer, x):
    """
    Calculate the Lipschitz constant for a given layer by computing the
    maximum singular value of the Jacobian.
    """
    y = layer(x)

    # Compute Jacobian by autograd
    jacobian = []
    for i in range(y.numel()):
        grad_output = torch.zeros_like(y)
        grad_output.view(-1)[i] = 1
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        jacobian.append(gradients.view(-1).detach().cpu().numpy())
    jacobian = torch.tensor(jacobian).view(y.numel(), x.numel())  # Construct Jacobian

    # Compute singular values and return the maximum value
    singular_values = torch.linalg.svdvals(jacobian)
    return singular_values.max().item()
