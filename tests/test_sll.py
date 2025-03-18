import pytest
import torch
from orthogonium.layers.conv.SLL import (
    SLLxAOCLipschitzResBlock,
    SDPBasedLipschitzResBlock,
    SDPBasedLipschitzDense,
    AOCLipschitzResBlock,
)


@pytest.mark.parametrize(
    "layer_class, init_params, batch_shape",
    [
        (
            SDPBasedLipschitzResBlock,
            {"cin": 4, "inner_dim_factor": 2, "kernel_size": 3},
            (1, 4, 8, 8),
        ),
        (
            SDPBasedLipschitzResBlock,
            {"cin": 2, "inner_dim_factor": 2, "kernel_size": 3, "groups": 2},
            (1, 2, 8, 8),
        ),
        (
            SLLxAOCLipschitzResBlock,
            {"cin": 4, "cout": 4, "inner_dim_factor": 2, "kernel_size": 3},
            (8, 4, 8, 8),
        ),
        (
            SLLxAOCLipschitzResBlock,
            {"cin": 4, "cout": 4, "inner_dim_factor": 2, "kernel_size": 3, "groups": 2},
            (8, 4, 8, 8),
        ),
        (
            SDPBasedLipschitzDense,
            {"in_features": 64, "out_features": 64, "inner_dim": 64},
            (8, 64),
        ),
        (
            AOCLipschitzResBlock,
            {"in_channels": 4, "inner_dim_factor": 2, "kernel_size": 3},
            (8, 4, 8, 8),
        ),
        (
            AOCLipschitzResBlock,
            {"in_channels": 4, "inner_dim_factor": 2, "kernel_size": 3, "groups": 2},
            (8, 4, 8, 8),
        ),
    ],
)
def test_lipschitz_layers(layer_class, init_params, batch_shape):
    """
    Generalized test for layers in the SLLx module to check Lipschitz constraints.
    """
    # Initialize layer
    layer = layer_class(**init_params)

    # Define input and target tensors
    x = torch.randn(*batch_shape, requires_grad=True)  # Input

    # Pre-optimization Lipschitz constant (if applicable)
    pre_lipschitz_constant = compute_lipschitz_constant(layer, x)
    print(f"{layer_class.__name__} | Before: {pre_lipschitz_constant:.6f}")
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
    print(f"{layer_class.__name__} | After: {post_lipschitz_constant:.6f}")
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
    # singular_values = torch.linalg.svdvals(jacobian)
    # return singular_values.max().item()
    return torch.linalg.matrix_norm(jacobian, ord=2).item()
