"""
Basically a copy of NEAT-Python's activation functions
https://github.com/CodeReclaimers/neat-python/blob/master/neat/activations.py
"""

import types
import jax.numpy as jnp
import jax
from typing import Callable


def sigmoid_activation(z):
    """Computes the sigmoid activation function.

    The sigmoid function squashes its input into a range between 0 and 1.
    It is commonly used in the output layer of binary classification problems.

    Args:
        z: The input array.

    Returns:
        The output array after applying the sigmoid function.

    Example:
        >>> import jax.numpy as jnp
        >>> sigmoid_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([0.11920291, 0.5       , 0.880797  ], dtype=float32)
    """
    z = jnp.clip(5.0 * z, -60.0, 60.0)
    return jax.nn.sigmoid(z)


def tanh_activation(z):
    """Computes the hyperbolic tangent (tanh) activation function.

    The tanh function squashes its input into a range between -1 and 1.
    It is often used in hidden layers of neural networks as it is zero-centered.

    Args:
        z: The input array.

    Returns:
        The output array after applying the tanh function.

    Example:
        >>> import jax.numpy as jnp
        >>> tanh_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([-0.90514827,  0.        ,  0.90514827], dtype=float32)
    """
    z = jnp.clip(2.5 * z, -60.0, 60.0)
    return jnp.tanh(z)


def sin_activation(z):
    """Computes the sine activation function.

    The sine function introduces periodicity into the network, which can be
    useful for tasks involving periodic patterns.

    Args:
        z: The input array.

    Returns:
        The output array after applying the sine function.

    Example:
        >>> import jax.numpy as jnp
        >>> sin_activation(jnp.array([0.0, jnp.pi/2, jnp.pi]))
        DeviceArray([0.0000000e+00, 1.0000000e+00, -8.7422777e-08], dtype=float32)
    """
    z = jnp.clip(5.0 * z, -60.0, 60.0)
    return jnp.sin(z)


def gauss_activation(z):
    """Computes the Gaussian activation function.

    The Gaussian function is a bell-shaped curve. It can be used to introduce
    radial basis function-like behavior in neural networks.

    Args:
        z: The input array.

    Returns:
        The output array after applying the Gaussian function.

    Example:
        >>> import jax.numpy as jnp
        >>> gauss_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([0.00673795, 1.        , 0.00673795], dtype=float32)
    """
    z = jnp.clip(z, -3.4, 3.4)
    return jnp.exp(-5.0 * z**2)


def relu_activation(z):
    """Computes the Rectified Linear Unit (ReLU) activation function.

    ReLU is a popular activation function in deep learning. It outputs the input
    directly if it is positive, and zero otherwise.

    Args:
        z: The input array.

    Returns:
        The output array after applying the ReLU function.

    Example:
        >>> import jax.numpy as jnp
        >>> relu_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([0., 0., 1.], dtype=float32)
    """
    return jax.nn.relu(z)


def identity_activation(z):
    """Computes the identity activation function.

    The identity function returns its input unchanged. It is often used in the
    output layer for regression tasks.

    Args:
        z: The input array.

    Returns:
        The input array.

    Example:
        >>> import jax.numpy as jnp
        >>> identity_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([-1.,  0.,  1.], dtype=float32)
    """
    return z


def clamped_activation(z):
    """Computes the clamped activation function.

    This function clips the input values to be within the range [-1.0, 1.0].

    Args:
        z: The input array.

    Returns:
        The output array with values clamped between -1.0 and 1.0.

    Example:
        >>> import jax.numpy as jnp
        >>> clamped_activation(jnp.array([-2.0, 0.5, 1.5]))
        DeviceArray([-1. ,  0.5,  1. ], dtype=float32)
    """
    return jnp.clip(z, -1.0, 1.0)


def inv_activation(z):
    """Computes the inverse activation function (1/z).

    This function computes the reciprocal of the input.
    Handles potential division by zero by returning 0.0 in such cases.

    Args:
        z: The input array.

    Returns:
        The output array after applying the inverse function.

    Example:
        >>> import jax.numpy as jnp
        >>> inv_activation(jnp.array([-2.0, 0.0, 2.0]))
        DeviceArray([-0.5,  0. ,  0.5], dtype=float32)
    """
    try:
        z = jnp.divide(1.0, z)
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    """Computes the natural logarithm activation function.

    This function computes the natural logarithm (base e) of the input.
    Inputs should be positive.

    Args:
        z: The input array. Must contain positive values.

    Returns:
        The output array after applying the natural logarithm.

    Example:
        >>> import jax.numpy as jnp
        >>> log_activation(jnp.array([1.0, jnp.e, 10.0]))
        DeviceArray([0.       , 0.9999999, 2.3025851], dtype=float32)
    """
    return jnp.log(z)


def exp_activation(z):
    """Computes the exponential activation function (e^z).

    This function computes e raised to the power of the input.

    Args:
        z: The input array.

    Returns:
        The output array after applying the exponential function.

    Example:
        >>> import jax.numpy as jnp
        >>> exp_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([0.36787945, 1.        , 2.7182817 ], dtype=float32)
    """
    z = jnp.clip(z, -60.0, 60.0)
    return jnp.exp(z)


def abs_activation(z):
    """Computes the absolute value activation function.

    This function returns the absolute value of each element in the input.

    Args:
        z: The input array.

    Returns:
        The output array with absolute values.

    Example:
        >>> import jax.numpy as jnp
        >>> abs_activation(jnp.array([-1.0, 0.0, 1.0]))
        DeviceArray([1., 0., 1.], dtype=float32)
    """
    return jnp.abs(z)


def hat_activation(z):
    """Computes the hat activation function.

    The hat function is defined as max(0, 1 - abs(z)). It creates a triangular
    shape centered at 0, with a maximum value of 1 and a width of 2.

    Args:
        z: The input array.

    Returns:
        The output array after applying the hat function.

    Example:
        >>> import jax.numpy as jnp
        >>> hat_activation(jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0]))
        DeviceArray([0. , 0.5, 1. , 0.5, 0. ], dtype=float32)
    """
    return jnp.maximum(0.0, 1 - abs(z))


def square_activation(z):
    """Computes the square activation function (z^2).

    This function squares each element in the input.

    Args:
        z: The input array.

    Returns:
        The output array with squared values.

    Example:
        >>> import jax.numpy as jnp
        >>> square_activation(jnp.array([-2.0, 0.0, 2.0]))
        DeviceArray([4., 0., 4.], dtype=float32)
    """
    return z**2


def cube_activation(z):
    """Computes the cube activation function (z^3).

    This function cubes each element in the input.

    Args:
        z: The input array.

    Returns:
        The output array with cubed values.

    Example:
        >>> import jax.numpy as jnp
        >>> cube_activation(jnp.array([-2.0, 0.0, 2.0]))
        DeviceArray([-8.,  0.,  8.], dtype=float32)
    """
    return z**3


class InvalidActivationFunction(TypeError):
    """Exception raised for invalid activation functions."""
    pass


def validate_activation(function):
    """Validates if the given object is a valid activation function.

    A valid activation function is a callable that accepts a single argument.

    Args:
        function: The object to validate.

    Raises:
        InvalidActivationFunction: If the function is not a callable or
            does not accept a single argument.
    """
    if not isinstance(
        function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)
    ):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    """Manages a collection of activation functions.

    This class provides a way to register, retrieve, and validate activation
    functions used within a CPPN (Compositional Pattern Producing Network).
    It initializes with a default set of common activation functions.

    Attributes:
        functions (dict): A dictionary mapping function names (str) to
            the actual function callables.

    Examples:
        >>> activation_set = ActivationFunctionSet()
        >>> sigmoid = activation_set.get("sigmoid")
        >>> print(sigmoid(jnp.array(0.0)))
        0.5
        >>> activation_set.is_valid("relu")
        True
        >>> def my_custom_activation(x):
        ...     return x * x * x
        >>> activation_set.add("custom_cube", my_custom_activation)
        >>> custom_cube = activation_set.get("custom_cube")
        >>> print(custom_cube(jnp.array(3.0)))
        27.0
    """

    def __init__(self):
        """Initializes the ActivationFunctionSet with a default set of functions."""
        self.functions = {}
        self.add("sigmoid", sigmoid_activation)
        self.add("tanh", tanh_activation)
        self.add("sin", sin_activation)
        self.add("gauss", gauss_activation)
        self.add("relu", relu_activation)
        self.add("identity", identity_activation)
        self.add("clamped", clamped_activation)
        self.add("inv", inv_activation)
        self.add("log", log_activation)
        self.add("exp", exp_activation)
        self.add("abs", abs_activation)
        self.add("hat", hat_activation)
        self.add("square", square_activation)
        self.add("cube", cube_activation)

    def add(self, name: str, function: Callable):
        """Adds a new activation function to the set.

        The provided function is validated to ensure it's a callable that
        accepts a single argument.

        Args:
            name: The name to register the function under.
            function: The activation function (a callable).

        Raises:
            InvalidActivationFunction: If the provided function is not valid.
        """
        validate_activation(function)
        self.functions[name] = function

    def get(self, name: str) -> Callable:
        """Retrieves an activation function by its name.

        Args:
            name: The name of the activation function to retrieve.

        Returns:
            The callable activation function.

        Raises:
            InvalidActivationFunction: If no function with the given name is found.
        """
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction(
                "No such activation function: {0!r}".format(name)
            )

        return f

    def is_valid(self, name: str) -> bool:
        """Checks if an activation function name exists in the set.

        Args:
            name: The name of the activation function to check.

        Returns:
            True if the function name is valid, False otherwise.
        """
        return name in self.functions
