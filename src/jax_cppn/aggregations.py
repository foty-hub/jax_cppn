import jax.numpy as jnp
import types
import warnings


def product_aggregation(x):
    # x: (N, c) -> returns (N, 1)
    return jnp.prod(x, axis=1, keepdims=False)


def sum_aggregation(x):
    return jnp.sum(x, axis=1, keepdims=False)


def max_aggregation(x):
    return jnp.max(x, axis=1, keepdims=False)


def min_aggregation(x):
    return jnp.min(x, axis=1, keepdims=False)


def maxabs_aggregation(x):
    # Get the index of the element with maximum absolute value in each batch.
    idx = jnp.argmax(jnp.abs(x), axis=1)
    # Use advanced indexing to select the max-abs values and reshape to (N, 1)
    return jnp.take_along_axis(x, idx[:, None], axis=1)


def median_aggregation(x):
    return jnp.median(x, axis=1, keepdims=False)


def mean_aggregation(x):
    return jnp.mean(x, axis=1, keepdims=False)


def no_aggregation(x):
    # If no aggregation is desired, x should already be (N, 1)
    if x.shape[1] != 1:
        raise ValueError("no_aggregation expects the input to have shape (N, 1)")
    return x


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):
    if not isinstance(
        function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)
    ):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1):
        raise InvalidAggregationFunction(
            "A function taking at least one argument is required"
        )


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add("product", product_aggregation)
        self.add("sum", sum_aggregation)
        self.add("max", max_aggregation)
        self.add("min", min_aggregation)
        self.add("maxabs", maxabs_aggregation)
        self.add("median", median_aggregation)
        self.add("mean", mean_aggregation)
        self.add("identity", no_aggregation)

    def add(self, name, function):
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction(
                "No such aggregation function: {0!r}".format(name)
            )

        return f

    def __getitem__(self, index):
        warnings.warn(
            "Use get, not indexing ([{!r}]), for aggregation functions".format(index),
            DeprecationWarning,
        )
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
