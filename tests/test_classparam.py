import pytest
from orthogonium.model_factory import ClassParam


# Dummy functions for testing
def function_1(a, b, c=3):
    return a, b, c


def function_with_kwargs(a, b=2, c=3):
    return a, b, c


def function_simple(a):
    return a


def test_classparam_init():
    """Test initialization of ClassParam."""
    cp = ClassParam(function_1, 1, 2, c=3)
    assert cp.fct == function_1
    assert cp.args == (1, 2)
    assert cp.kwargs == {"c": 3}


def test_classparam_call():
    """Test calling ClassParam with default args and overwrites."""
    # Initialize with defaults
    cp = ClassParam(function_1, 1, 2, c=3)

    # Test default behavior
    assert cp() == (1, 2, 3)

    # Test positional overwrite
    assert cp(4) == (4, 2, 3)

    # Test keyword overwrite
    assert cp(4, c=5) == (4, 2, 5)

    # Test both positional and keyword overwrites
    assert cp(6, 7, c=8) == (6, 7, 8)


def test_classparam_call_with_kwargs_only():
    """Test keyword-based overrides when no positional args are provided."""
    cp = ClassParam(function_with_kwargs, 1, b=5)

    # Default behavior
    assert cp() == (1, 5, 3)

    # Override keyword-only arg
    assert cp(c=10) == (1, 5, 10)

    # Override positional and keyword args
    assert cp(6, b=7, c=9) == (6, 7, 9)


def test_classparam_call_no_function():
    """Test calling a ClassParam instance without a function."""
    cp = ClassParam()
    assert callable(cp)
    identity_function = cp()
    assert callable(identity_function)
    assert identity_function(42) == 42  # Should behave like an identity function


def test_classparam_str():
    """Test string representation of ClassParam."""
    cp = ClassParam(function_1, 1, 2, c=3)
    # Check string interpolation
    assert str(cp) == "function_1(1,2,c=3)"


def test_classparam_get_config():
    """Test get_config generates the correct dict representation."""
    cp = ClassParam(function_1, 1, 2, c=3)

    # Nested config without flattening
    config = cp.get_config(flatten=False)
    assert config == {
        "fct": "function_1",
        "args": {"args_0": "1", "args_1": "2"},
        "kwargs": {"c": "3"},
    }

    # Flatten the config
    flat_config = cp.get_config(flatten=True)
    assert flat_config == {
        "fct": "function_1",
        "args_0": "1",
        "args_1": "2",
        "c": "3",
    }


def test_classparam_overwrite_restrictions():
    """Test that positional args cannot be overwritten by kwargs."""
    cp = ClassParam(function_1, 1, 2, c=3)

    with pytest.raises(TypeError):
        cp(b="new_value")  # Overwriting positional args with kwargs should fail


def test_flatten_config():
    """Test the flatten_config utility method."""
    child_cp = ClassParam(function_simple, 5)
    parent_cp = ClassParam(function_1, child_cp, 10, c=7)

    flattened = parent_cp.get_config(flatten=True)
    assert flattened == {
        "fct": "function_1",
        "args_0/fct": "function_simple",
        "args_0/args_0": "5",
        "args_1": "10",
        "c": "7",
    }
