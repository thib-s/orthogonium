import pytest
import torch

from orthogonium.layers import LayerCentering2D, BatchCentering2D


@pytest.mark.parametrize(
    "layer_fn, num_features, orthogonal",
    [
        (lambda: LayerCentering2D(num_features=4), 4, False),
        (lambda: BatchCentering2D(num_features=4), 4, True),
    ],
)
@pytest.mark.parametrize(
    "mean, std",
    [
        (0, 1),  # Standard Normal Distribution
        (5, 2),  # Higher mean and variance
        (-3, 1),  # Shifted mean
        (0, 0.1),  # Low variance
        (10, 5),  # Very high variance
    ],
)
def test_lipschitz_constant_with_various_distributions(
    layer_fn, num_features, orthogonal, mean, std
):
    """
    Test if the layer satisfies the Lipschitz property when the input
    comes from distributions with different means and variances.
    """
    layer = layer_fn()
    layer.train()  # Set layer to training mode

    batch_size, h, w = 8, 8, 8  # Input dimensions

    # Generate input tensor from a specific distribution
    x = torch.randn(batch_size, num_features, h, w) * std + mean
    x.requires_grad_(True)  # Enable gradient tracking

    # Forward pass through the layer
    y = layer(x)

    # Calculate Jacobian
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
    jacobian = torch.tensor(jacobian).view(
        y.numel(), x.numel()
    )  # Construct Jacobian matrix

    # Validate Lipschitz constant
    singular_values = torch.linalg.svdvals(jacobian)
    assert singular_values.max() <= 1 + 1e-4, (
        f"Lipschitz constraint violated for input distribution with mean={mean}, std={std}; "
        f"max singular value: {singular_values.max()}"
    )
    if orthogonal:
        assert (
            singular_values.min() >= 1 - 1e-4
        ), f"Orthogonality constraint violated for input distribution with mean={mean}, std={std}; "
