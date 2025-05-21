from jax_cppn.activations import ActivationFunctionSet
from jax_cppn.aggregations import AggregationFunctionSet
import jax.numpy as jnp # Add jax.numpy for examples

# Get function sets for activations and aggregations.
activations = ActivationFunctionSet()
aggregations = AggregationFunctionSet()


class Node:
    """Represents a single node in a CPPN.

    Each node has an activation function and an aggregation function. It processes
    inputs by first aggregating them and then applying the activation function.

    Class Attributes:
        activation (Callable): The activation function (e.g., sigmoid, relu).
        aggregation (Callable): The aggregation function (e.g., sum, product).
        node_id (int): A unique identifier for the node.
        label (str | None): An optional label for the node (e.g., for visualization).
        act_str (str): The string identifier for the activation function.
        agg_str (str): The string identifier for the aggregation function.

    Example:
        >>> import jax.numpy as jnp
        >>> from jax_cppn.node import Node
        >>> node = Node(activation="sigmoid", aggregation="sum", node_id=0, label="H0")
        >>> inputs = jnp.array([[1.0, 0.5], [-0.5, 0.0]]) # Example inputs for aggregation
        >>> output = node(inputs)
        >>> print(output)
        [0.81757444 0.37754068]
    """
    def __init__(
        self, activation: str, aggregation: str, node_id: int, label: str | None = None
    ):
        """Initializes a Node.

        Args:
            activation: The name of the activation function to use (e.g., "sigmoid").
                This function is retrieved from the global `activations` set.
            aggregation: The name of the aggregation function to use (e.g., "sum").
                This function is retrieved from the global `aggregations` set.
            node_id: A unique integer identifier for this node.
            label: An optional string label for the node (e.g., "Hidden_1", "Input_X").
        """
        self.act_str = activation
        self.agg_str = aggregation
        self.activation = activations.get(activation)
        self.aggregation = aggregations.get(aggregation)
        self.node_id = node_id
        self.label = label

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Processes input signals through aggregation and activation.

        The input array is first passed to the node's aggregation function.
        The result of the aggregation is then passed to the node's activation function.

        Args:
            inputs: A JAX numpy array. The shape is typically (N, c), where N is
                the batch size (number of patterns to evaluate) and c is the
                number of incoming connections to this node. The aggregation
                function will reduce this to (N,).

        Returns:
            A JAX numpy array of shape (N,) representing the node's output after
            aggregation and activation.
        """
        z = self.aggregation(inputs)
        z = self.activation(z)
        return z


# Define an input node type that simply passes its external input
class InputNode(Node):
    """A specialized Node that represents an input to the CPPN.

    Input nodes use an "identity" activation function and an "identity"
    aggregation function. This means they effectively pass through the
    value provided to them without modification by these steps. Their primary
    role is to serve as entry points for external data into the network.

    Class Attributes:
        Inherits all attributes from `Node`. `act_str` is "identity" and
        `agg_str` is "identity".

    Example:
        >>> from jax_cppn.node import InputNode
        >>> input_node = InputNode(node_id=1, label="X_coord")
        >>> print(f"Label: {input_node.label}, ID: {input_node.node_id}")
        Label: X_coord, ID: 1
        >>> print(f"Activation: {input_node.act_str}, Aggregation: {input_node.agg_str}")
        Activation: identity, Aggregation: identity
        >>> # InputNodes expect a (N, 1) input for their __call__ method
        >>> example_input = jnp.array([[0.5], [-0.2]])
        >>> output = input_node(example_input)
        >>> print(output)
        [ 0.5 -0.2]
    """
    def __init__(self, node_id: int, label: str | None = None):
        """Initializes an InputNode.

        Args:
            node_id: A unique integer identifier for this input node.
            label: An optional string label for the node (e.g., "X", "Y").
        """
        super().__init__(
            activation="identity", aggregation="identity", node_id=node_id, label=label
        )


class OutputNode(Node):
    """A specialized Node that represents an output of the CPPN.

    Output nodes use an "identity" activation function by default, meaning they
    typically do not apply a final non-linearity after aggregation unless
    a different activation were set (which is not the default for this class).
    They aggregate incoming signals using a specified aggregation function.

    Class Attributes:
        Inherits all attributes from `Node`. `act_str` is "identity".
        `agg_str` can be specified during initialization.

    Example:
        >>> from jax_cppn.node import OutputNode
        >>> # Output node using 'sum' aggregation
        >>> output_node_sum = OutputNode(node_id=2, aggregation="sum", label="Output_Sum")
        >>> print(f"Label: {output_node_sum.label}, Activation: {output_node_sum.act_str}, Aggregation: {output_node_sum.agg_str}")
        Label: Output_Sum, Activation: identity, Aggregation: sum
        >>>
        >>> # Output node using 'mean' aggregation
        >>> output_node_mean = OutputNode(node_id=3, aggregation="mean", label="Output_Mean")
        >>> print(f"Label: {output_node_mean.label}, Activation: {output_node_mean.act_str}, Aggregation: {output_node_mean.agg_str}")
        Label: Output_Mean, Activation: identity, Aggregation: mean
        >>>
        >>> # Example of calling an OutputNode
        >>> import jax.numpy as jnp
        >>> inputs_for_output = jnp.array([[0.8, 0.6, 1.0], [0.2, 0.3, 0.1]]) # (batch_size, num_connections)
        >>> result = output_node_sum(inputs_for_output)
        >>> print(result)
        [2.4 0.6]
    """
    def __init__(self, node_id: int, aggregation: str = "sum", label: str | None = None):
        """Initializes an OutputNode.

        Args:
            node_id: A unique integer identifier for this output node.
            aggregation: The name of the aggregation function to use (e.g., "sum", "mean").
                Defaults to "sum".
            label: An optional string label for the node (e.g., "Color_R", "Value").
        """
        super().__init__(
            activation="identity", aggregation=aggregation, node_id=node_id, label=label
        )
