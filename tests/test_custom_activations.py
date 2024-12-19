import pytest
import torch
from orthogonium.layers.custom_activations import (
    Abs,
    MaxMin,
    HouseHolder,
    HouseHolder_Order_2,
)


# ------------------------- Abs Tests -------------------------
def test_abs_output():
    abs_layer = Abs()
    x = torch.tensor([-1.0, 0.0, 1.0, -2.0])
    y = abs_layer(x)
    assert torch.allclose(
        y, torch.tensor([1.0, 0.0, 1.0, 2.0])
    ), "Abs function is incorrect"


def test_abs_shapes():
    abs_layer = Abs()
    x = torch.randn(2, 3, 4, 5)  # Random 4D tensor
    y = abs_layer(x)
    assert y.shape == x.shape, "Output shape mismatch for Abs function"


# ------------------------- MaxMin Tests -------------------------
def test_maxmin_output_shapes():
    maxmin_layer = MaxMin(axis=1)
    x = torch.randn(2, 6, 4, 4)
    y = maxmin_layer(x)
    assert y.shape == x.shape, "Output shape mismatch for MaxMin"


def test_maxmin_absorbant():
    maxmin_layer = MaxMin(axis=1)
    x = torch.randn(2, 6, 4, 4)
    y = maxmin_layer(x)
    z = maxmin_layer(y)
    assert torch.allclose(
        y.cpu(), z.cpu()
    ), "MaxMin layer is not idempotent (output of MaxMin is not input)"


def test_maxmin_invalid_input():
    maxmin_layer = MaxMin(axis=1)
    x = torch.randn(2, 5)  # Odd-sized dimension
    with pytest.raises(ValueError):
        maxmin_layer(x)


# ------------------------- HouseHolder Tests -------------------------
def test_householder_shapes():
    layer = HouseHolder(channels=4)
    x = torch.randn(2, 4, 8, 8)  # Input channels divisible by 2
    y = layer(x)
    assert y.shape == x.shape, "Output shape mismatch for HouseHolder"


def test_householder_gradients():
    layer = HouseHolder(channels=4)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None, "Gradients not computed for HouseHolder"


def test_householder_invalid_channels():
    with pytest.raises(AssertionError):
        HouseHolder(channels=3)  # Channels not divisible by 2


# ------------------------- HouseHolder_Order_2 Tests -------------------------
def test_householder_order2_shapes():
    layer = HouseHolder_Order_2(channels=4)
    x = torch.randn(2, 4, 8, 8)
    y = layer(x)
    assert y.shape == x.shape, "Output shape mismatch for HouseHolder_Order_2"


def test_householder_order2_gradients():
    layer = HouseHolder_Order_2(channels=6)
    x = torch.randn(2, 6, 16, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None, "Gradients not computed for HouseHolder_Order_2"


def test_householder_order2_invalid_channels():
    with pytest.raises(AssertionError):
        HouseHolder_Order_2(channels=5)  # Odd number of channels


# ------------------------- Lipschitz Property Tests -------------------------
@pytest.mark.parametrize(
    "layer_fn, channels",
    [
        (Abs, None),
        (lambda: MaxMin(axis=1), None),
        (lambda: HouseHolder(channels=4), 4),
        (lambda: HouseHolder_Order_2(channels=4), 4),
    ],
)
def test_lipschitz_property(layer_fn, channels):
    """
    Tests if the layer satisfies the 1-Lipschitz property.
    """
    if callable(layer_fn):
        layer = layer_fn()
    else:
        layer = layer_fn

    channels = channels or 4  # Default to 4 channels if unspecified
    x = torch.randn(2, channels, requires_grad=True)
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
    jacobian = torch.tensor(jacobian).view(y.numel(), x.numel())  # Construct Jacobian

    # Compute singular values and check Lipschitz property
    singular_values = torch.linalg.svdvals(jacobian)
    assert (
        singular_values.max() <= 1 + 1e-4
    ), f"{layer.__class__.__name__} is not 1-Lipschitz"
    assert (
        singular_values.min() >= 1 - 1e-4
    ), f"{layer.__class__.__name__} is not orthogonal"


# ------------------------- Run All Tests -------------------------
if __name__ == "__main__":
    pytest.main()
