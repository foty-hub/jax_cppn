# import jax
# import jax.numpy as jnp
from typing import Callable

from jax_cppn.activations import ActivationFunctionSet
from jax_cppn.aggregations import AggregationFunctionSet

# Get function sets for activations and aggregations.
activations = ActivationFunctionSet()
aggregations = AggregationFunctionSet()


class Node:
    def __init__(
        self, activation: str, aggregation: str, node_id: int, label: str | None = None
    ):
        self.act_str = activation
        self.agg_str = aggregation
        self.activation = activations.get(activation)
        self.aggregation = aggregations.get(aggregation)
        self.node_id = node_id
        self.label = label

    def __call__(self, inputs):
        z = self.aggregation(inputs)
        z = self.activation(z)
        return z


# Define an input node type that simply passes its external input
class InputNode(Node):
    def __init__(self, node_id: int, label: str | None = None):
        super().__init__(
            activation="identity", aggregation="identity", node_id=node_id, label=label
        )
