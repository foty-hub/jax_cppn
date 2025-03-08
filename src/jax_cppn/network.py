# %%
import jax.numpy as jnp
import jax
from jax_cppn.activations import ActivationFunctionSet
from jax_cppn.aggregations import AggregationFunctionSet
from jax_cppn.node import Node

# %%
# Get function sets for activations and aggregations.
activations = ActivationFunctionSet()
aggregations = AggregationFunctionSet()


# Define an input node type that simply passes its external input
class InputNode(Node):
    def __init__(self, node_id: int):
        super().__init__(
            activation=activations.get("identity"),
            aggregation=aggregations.get("identity"),
            node_id=node_id,
        )


# A simple connection structure: each connection carries a weight from one node to another.
class Connection:
    def __init__(self, in_node: int, out_node: int, weight: float):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight

    def __repr__(self):
        return f"Connection({self.in_node} -> {self.out_node}, weight={self.weight})"


# The CPPN network class: it holds a collection of nodes and connections,
# builds a graph structure, and computes a forward pass given input values.
class CPPN:
    def __init__(self, nodes: list[Node], connections):
        """
        nodes: list of Node objects. Each node must have a unique node_id.
        connections: list of Connection objects.
        """
        # Map node_id to node instance
        self.nodes = {node.node_id: node for node in nodes}
        self.connections = connections

        # Build mapping from each node id to its incoming connections.
        self.incoming = {node_id: [] for node_id in self.nodes}
        for conn in connections:
            self.incoming[conn.out_node].append(conn)

        # Determine evaluation order by topologically sorting the nodes.
        self.topo_order = self._topological_sort()

    def _topological_sort(self):
        # Kahn's algorithm: count incoming edges and repeatedly remove nodes with zero in-degree.
        in_degree = {node_id: 0 for node_id in self.nodes}
        for conn in self.connections:
            in_degree[conn.out_node] += 1

        # Start with nodes that have no incoming connections (usually input nodes).
        queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            current = queue.pop(0)
            topo_order.append(current)
            # Find all outgoing connections from the current node.
            for conn in self.connections:
                if conn.in_node == current:
                    in_degree[conn.out_node] -= 1
                    if in_degree[conn.out_node] == 0:
                        queue.append(conn.out_node)

        if len(topo_order) != len(self.nodes):
            raise ValueError(
                "Graph has cycles or disconnected parts; topological sort failed."
            )
        return topo_order

    def __call__(self, input_values: dict):
        """
        Compute the forward pass through the network.

        input_values: a dictionary mapping input node_id to a jax.numpy array value.
                      All non-input nodes will be computed from their incoming connections.

        Returns a dictionary mapping each node_id to its computed output.
        """
        computed = {}

        # Process nodes in topologically sorted order.
        for node_id in self.topo_order:
            node = self.nodes[node_id]

            # For input nodes, take the externally provided value.
            if node_id in input_values:
                computed[node_id] = input_values[node_id]
            else:
                # For non-input nodes, gather weighted inputs from all incoming connections.
                inputs = []
                for conn in self.incoming[node_id]:
                    # Multiply the output from the source node by the connection's weight.
                    inputs.append(conn.weight * computed[conn.in_node])
                # computed[node_id] = node(jnp.array(inputs))
                input_array = jnp.stack(inputs, axis=1)
                print("-" * 50)
                print(f"handling node {node_id}")
                print(f"{input_array.shape=}")
                agg = node.aggregation(input_array)
                print(f"{agg.shape=}")
                act = node.activation(agg)
                print(f"{act.shape=}")
                computed[node_id] = act
        return computed

    def __repr__(self):
        # Build a string listing all nodes.
        nodes_str = "\n".join(
            [
                f"  Node {node_id}: {node}"
                for node_id, node in sorted(self.nodes.items())
            ]
        )
        # Build a string listing all connections.
        connections_str = "\n".join([f"  {conn}" for conn in self.connections])
        return (
            f"CPPN Network:\n"
            f"Nodes:\n{nodes_str}\n\n"
            f"Connections:\n{connections_str}\n\n"
            f"Topological Order: {self.topo_order}"
        )

    def __str__(self):
        return self.__repr__()


# %%
if __name__ == "__main__":
    # Create input nodes.
    input_node0 = InputNode(0)
    input_node1 = InputNode(1)

    # Create three hidden nodes, each with a different activation and aggregation.
    hidden_node1 = Node(
        activation=activations.get("sigmoid"),
        aggregation=aggregations.get("sum"),
        node_id=2,
    )
    hidden_node2 = Node(
        activation=activations.get("tanh"),
        aggregation=aggregations.get("product"),
        node_id=3,
    )
    hidden_node3 = Node(
        activation=activations.get("relu"),
        aggregation=aggregations.get("mean"),
        node_id=4,
    )

    # Create an output node. For this example, we use identity activation and max aggregation.
    output_node = Node(
        activation=activations.get("identity"),
        aggregation=aggregations.get("max"),
        node_id=5,
    )

    # List all nodes in the network.
    nodes = [
        input_node0,
        input_node1,
        hidden_node1,
        hidden_node2,
        hidden_node3,
        output_node,
    ]

    # Create connections:
    # Hidden node 1 (node_id=2) receives inputs from both input nodes.
    connections = [
        Connection(in_node=0, out_node=2, weight=0.6),
        Connection(in_node=1, out_node=2, weight=0.4),
        # Hidden node 2 (node_id=3) receives inputs from both input nodes with different weights.
        Connection(in_node=0, out_node=3, weight=0.8),
        Connection(in_node=1, out_node=3, weight=-0.5),
        # Hidden node 3 (node_id=4) also receives inputs from both input nodes.
        Connection(in_node=0, out_node=4, weight=0.3),
        Connection(in_node=1, out_node=4, weight=0.7),
        # The output node (node_id=5) aggregates outputs from all three hidden nodes.
        Connection(in_node=2, out_node=5, weight=1.0),
        Connection(in_node=3, out_node=5, weight=0.5),
        Connection(in_node=4, out_node=5, weight=-0.2),
    ]

    # Build the CPPN network.
    cppn_net = CPPN(nodes, connections)

    # Prepare example input data.
    # Let's assume a batch size of 100. Each input should be of shape (100, 1).
    x_coords = jnp.linspace(-5, 5, 100)
    # For a second input, we can use a sine transformation.
    y_coords = jnp.sin(x_coords)

    inputs = {0: x_coords, 1: y_coords}

    # Perform the forward pass.
    outputs = cppn_net(inputs)

    # Plot the output.
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x_coords, outputs[5])
    plt.xlabel("x")
    plt.ylabel("output")
    plt.title("Complex CPPN Network Output")
    plt.show()
# %%
