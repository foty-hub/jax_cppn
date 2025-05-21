import jax.numpy as jnp
import types
import warnings
from typing import Callable


def product_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by computing their product.

    This function calculates the product of elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the product of inputs for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 0.5]]) # Shape (2, 3)
        >>> product_aggregation(x)
        DeviceArray([ 6., 10.], dtype=float32)
    """
    # x: (N, c) -> returns (N,)
    return jnp.prod(x, axis=1, keepdims=False)


def sum_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by computing their sum.

    This function calculates the sum of elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the sum of inputs for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Shape (2, 3)
        >>> sum_aggregation(x)
        DeviceArray([ 6., 15.], dtype=float32)
    """
    return jnp.sum(x, axis=1, keepdims=False)


def max_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by selecting the maximum value.

    This function finds the maximum value among elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the maximum input for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]) # Shape (2, 3)
        >>> max_aggregation(x)
        DeviceArray([5., 6.], dtype=float32)
    """
    return jnp.max(x, axis=1, keepdims=False)


def min_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by selecting the minimum value.

    This function finds the minimum value among elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the minimum input for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, -2.0, 3.0], [4.0, 0.5, 6.0]]) # Shape (2, 3)
        >>> min_aggregation(x)
        DeviceArray([-2. ,  0.5], dtype=float32)
    """
    return jnp.min(x, axis=1, keepdims=False)


def maxabs_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by selecting the value with the maximum absolute magnitude.

    This function identifies the element with the largest absolute value along
    the second axis (axis=1) for each item in the batch and returns the
    original value (not its absolute).

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the input with the maximum
        absolute value for each item in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, -5.0, 3.0], [-4.0, 2.0, -6.0]]) # Shape (2, 3)
        >>> result = maxabs_aggregation(x) # result will be shape (2,)
        >>> print(result)
        [-5. -6.]
    """
    # Get the index of the element with maximum absolute value in each batch.
    idx = jnp.argmax(jnp.abs(x), axis=1)
    # Use advanced indexing to select the max-abs values and reshape to (N,)
    # The result of take_along_axis will be (N,1), so we squeeze it.
    return jnp.take_along_axis(x, idx[:, None], axis=1).squeeze(axis=1)


def median_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by computing their median.

    This function calculates the median of elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the median of inputs for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 10.0], [4.0, 5.0, 0.0]]) # Shape (2, 3)
        >>> median_aggregation(x)
        DeviceArray([2., 4.], dtype=float32)
    """
    return jnp.median(x, axis=1, keepdims=False)


def mean_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Aggregates inputs by computing their mean.

    This function calculates the mean of elements along the second axis (axis=1)
    of the input array.

    Args:
        x: A JAX array of shape (N, c), where N is the batch size and c is
           the number of incoming connections.

    Returns:
        A JAX array of shape (N,) representing the mean of inputs for each item
        in the batch.

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # Shape (2, 3)
        >>> mean_aggregation(x)
        DeviceArray([2., 5.], dtype=float32)
    """
    return jnp.mean(x, axis=1, keepdims=False)


def no_aggregation(x: jnp.ndarray) -> jnp.ndarray:
    """Represents no aggregation, returning the input as is, after validation.

    This function expects the input `x` to already be aggregated, i.e.,
    have a shape of (N, 1). If the shape is different, it raises a ValueError.
    It then squeezes the last dimension to return an array of shape (N,).

    Args:
        x: A JAX array, expected to be of shape (N, 1).

    Returns:
        A JAX array of shape (N,).

    Raises:
        ValueError: If the input array `x` does not have shape (N, 1).

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.array([[1.0], [4.0]]) # Shape (2, 1)
        >>> no_aggregation(x)
        DeviceArray([1., 4.], dtype=float32)
        >>> y = jnp.array([[1.0, 2.0]]) # Shape (1, 2)
        >>> try:
        ...   no_aggregation(y)
        ... except ValueError as e:
        ...   print(e)
        no_aggregation expects the input to have shape (N, 1)
    """
    if x.shape[1] != 1:
        raise ValueError("no_aggregation expects the input to have shape (N, 1)")
    return x.squeeze(axis=1) # Squeeze to make it (N,)


class InvalidAggregationFunction(TypeError):
    """Exception raised for invalid aggregation functions."""
    pass


def validate_aggregation(function: Callable):
    """Validates if the given object is a valid aggregation function.

    A valid aggregation function is a callable that accepts at least one argument.

    Args:
        function: The object to validate.

    Raises:
        InvalidAggregationFunction: If the function is not a callable or
            does not accept at least one argument.
    """
    if not isinstance(
        function, (types.BuiltinFunctionType, types.FunctionType, types.LambdaType)
    ):
        raise InvalidAggregationFunction("A function object is required.")

    if not (function.__code__.co_argcount >= 1): # `inspect` is preferred but adds dependency
        raise InvalidAggregationFunction(
            "A function taking at least one argument is required"
        )


class AggregationFunctionSet(object):
    """Manages a collection of aggregation functions.

    This class provides a way to register, retrieve, and validate aggregation
    functions. These functions are typically used in neural networks to combine
    multiple input signals into a single value before applying an activation function.
    It initializes with a default set of common aggregation functions.

    Attributes:
        functions (dict): A dictionary mapping function names (str) to
            the actual function callables.

    Examples:
        >>> agg_set = AggregationFunctionSet()
        >>> sum_func = agg_set.get("sum")
        >>> data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> print(sum_func(data))
        [3. 7.]
        >>> agg_set.is_valid("mean")
        True
        >>> def my_custom_agg(x):
        ...     return jnp.min(x, axis=1) * 2
        >>> agg_set.add("custom_double_min", my_custom_agg)
        >>> custom_func = agg_set.get("custom_double_min")
        >>> print(custom_func(data))
        [2. 6.]
    """

    def __init__(self):
        """Initializes the AggregationFunctionSet with a default set of functions."""
        self.functions = {}
        self.add("product", product_aggregation)
        self.add("sum", sum_aggregation)
        self.add("max", max_aggregation)
        self.add("min", min_aggregation)
        self.add("maxabs", maxabs_aggregation)
        self.add("median", median_aggregation)
        self.add("mean", mean_aggregation)
        self.add("identity", no_aggregation) # 'identity' is a common alias for no_aggregation

    def add(self, name: str, function: Callable):
        """Adds a new aggregation function to the set.

        The provided function is validated to ensure it's a callable that
        accepts at least one argument.

        Args:
            name: The name (str) to register the function under.
            function: The aggregation function (a callable).

        Raises:
            InvalidAggregationFunction: If the provided function is not valid.
        """
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name: str) -> Callable:
        """Retrieves an aggregation function by its name.

        Args:
            name: The name (str) of the aggregation function to retrieve.

        Returns:
            The callable aggregation function.

        Raises:
            InvalidAggregationFunction: If no function with the given name is found.
        """
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction(
                "No such aggregation function: {0!r}".format(name)
            )
        return f

    def __getitem__(self, index: str) -> Callable:
        """Provides dictionary-like access to aggregation functions (deprecated).

        Args:
            index: The name of the aggregation function.

        Returns:
            The callable aggregation function.

        Raises:
            DeprecationWarning: Always, as this method is deprecated.
            InvalidAggregationFunction: If the function name is not found.
        """
        warnings.warn(
            "Use get, not indexing ([{!r}]), for aggregation functions".format(index),
            DeprecationWarning,
            stacklevel=2, # Show warning at the caller's location
        )
        return self.get(index)

    def is_valid(self, name: str) -> bool:
        """Checks if an aggregation function name exists in the set.

        Args:
            name: The name (str) of the aggregation function to check.

        Returns:
            True if the function name is valid (exists in the set), False otherwise.
        """
        return name in self.functions
