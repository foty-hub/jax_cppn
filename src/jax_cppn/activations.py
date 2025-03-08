"""
Basically a copy of NEAT-Python's activation functions
https://github.com/CodeReclaimers/neat-python/blob/master/neat/activations.py
"""

import math
import types
import jax.numpy as jnp
import jax


def sigmoid_activation(z):
    z = jnp.clip(5.0 * z, -60.0, 60.0)
    return jax.nn.sigmoid(z)


def tanh_activation(z):
    z = jnp.clip(2.5 * z, -60.0, 60.0)
    return jnp.tanh(z)


def sin_activation(z):
    z = jnp.clip(5.0 * z, -60.0, 60.0)
    return jnp.sin(z)


def gauss_activation(z):
    z = jnp.clip(z, -3.4, 3.4)
    return jnp.exp(-5.0 * z**2)


def relu_activation(z):
    return jax.nn.relu(z)


def identity_activation(z):
    return z


def clamped_activation(z):
    return jnp.clip(z, -1.0, 1.0)


def inv_activation(z):
    try:
        z = jnp.divide(1.0, z)
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    return jnp.log(z)


def exp_activation(z):
    z = jnp.clip(z, -60.0, 60.0)
    return jnp.exp(z)


def abs_activation(z):
    return jnp.abs(z)


def hat_activation(z):
    return jnp.maximum(0.0, 1 - abs(z))


def square_activation(z):
    return z**2


def cube_activation(z):
    return z**3


class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    if not isinstance(
        function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)
    ):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
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

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction(
                "No such activation function: {0!r}".format(name)
            )

        return f

    def is_valid(self, name):
        return name in self.functions
