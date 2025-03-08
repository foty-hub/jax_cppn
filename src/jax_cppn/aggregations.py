import jax.numpy as jnp
import types
import warnings


def product_aggregation(x):
    arr = jnp.array(x)
    return jnp.prod(arr)


def sum_aggregation(x):
    arr = jnp.array(x)
    return jnp.sum(arr)


def max_aggregation(x):
    arr = jnp.array(x)
    return jnp.max(arr)


def min_aggregation(x):
    arr = jnp.array(x)
    return jnp.min(arr)


def maxabs_aggregation(x):
    arr = jnp.array(x)
    # Find the index of the element with the maximum absolute value
    index = jnp.argmax(jnp.abs(arr))
    return arr[index]


def median_aggregation(x):
    arr = jnp.array(x)
    return jnp.median(arr)


def mean_aggregation(x):
    arr = jnp.array(x)
    return jnp.mean(arr)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):  # TODO: Recognize when need `reduce`
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
