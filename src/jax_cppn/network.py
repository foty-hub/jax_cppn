# %%
import jax
import random
import jax.numpy as jnp
from jax_cppn.node import Node, InputNode
from jax_cppn.vis import visualize_cppn_network, plot_output

# %%
PERMITTED_MUTATIONS = ["gauss", "sin", "sigmoid", "tanh"]


# A simple connection structure: each connection carries a weight from one node to another.
class Connection:
    def __init__(self, in_node: int, out_node: int, weight: float) -> None:
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight

    def __repr__(self):
        return f"Connection({self.in_node} -> {self.out_node}, weight={self.weight})"


# The CPPN network class: it holds a collection of nodes and connections,
# builds a graph structure, and computes a forward pass given input values.
class CPPN:
    def __init__(self, nodes: list, connections) -> None:
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

    def rebuild_graph(self):
        """Rebuild the incoming connections mapping and re-compute topological order."""
        self.incoming = {node_id: [] for node_id in self.nodes}
        for conn in self.connections:
            self.incoming[conn.out_node].append(conn)
        self.topo_order = self._topological_sort()

    def __call__(self, input_values: dict):
        self.computed = {}
        output_id = -1
        # Process nodes in topologically sorted order.
        for node_id in self.topo_order:
            output_id = node_id
            node = self.nodes[node_id]

            # For input nodes, take the externally provided value.
            if node_id in input_values:
                self.computed[node_id] = input_values[node_id]
            else:
                # For non-input nodes, gather weighted inputs from all incoming connections.
                inputs = []
                for conn in self.incoming[node_id]:
                    inputs.append(conn.weight * self.computed[conn.in_node])
                input_array = jnp.stack(inputs, axis=1)
                agg = node.aggregation(input_array)
                act = node.activation(agg)
                self.computed[node_id] = act
        return self.computed[output_id]

    def __repr__(self):
        nodes_str = "\n".join(
            [
                f"  Node {node_id}: {node}"
                for node_id, node in sorted(self.nodes.items())
            ]
        )
        connections_str = "\n".join([f"  {conn}" for conn in self.connections])
        return (
            f"CPPN Network:\n"
            f"Nodes:\n{nodes_str}\n\n"
            f"Connections:\n{connections_str}\n\n"
            f"Topological Order: {self.topo_order}"
        )

    def __str__(self):
        return self.__repr__()

    # --- Mutation operators below ---

    def mutate_add_node(self, connection_index: int = None):
        """
        Mutate the network by inserting a new node in the middle of an existing connection.
        If no connection_index is provided, one is chosen at random.
        The selected connection is removed and replaced with two new connections.
        """
        if not self.connections:
            print("No connections available to split.")
            return

        if connection_index is None:
            connection_index = random.randrange(len(self.connections))
        old_conn = self.connections.pop(connection_index)

        # Create a new node with a new unique node id.
        new_node_id = max(self.nodes.keys()) + 1
        # Here we choose default activation and aggregation; you may randomize these if desired.
        new_activation = random.choice(PERMITTED_MUTATIONS)
        label = r"$\sigma$" if new_activation == "sigmoid" else None
        new_node = Node(
            activation=new_activation,
            aggregation="sum",
            node_id=new_node_id,
            label=label,
        )
        self.nodes[new_node_id] = new_node

        # Create two new connections:
        # Connection from original in_node to new node.
        conn1 = Connection(in_node=old_conn.in_node, out_node=new_node_id, weight=1.0)
        # Connection from new node to original out_node, inheriting the old connection's weight.
        conn2 = Connection(
            in_node=new_node_id, out_node=old_conn.out_node, weight=old_conn.weight
        )
        self.connections.extend([conn1, conn2])

        self.rebuild_graph()

    def mutate_add_connection(self, in_node: int, out_node: int, weight: float = None):
        """
        Mutate the network by adding a new connection from in_node to out_node.
        Checks for duplicate connections and potential cycles.
        If weight is not provided, a random weight in the range [-1, 1] is assigned.
        """
        if in_node not in self.nodes or out_node not in self.nodes:
            raise ValueError("Invalid node ids provided for connection.")

        # Check if the connection already exists.
        for conn in self.connections:
            if conn.in_node == in_node and conn.out_node == out_node:
                print("Connection already exists.")
                return

        # Check for cycle creation by performing a DFS from out_node to see if we can reach in_node.
        def has_path(current, target, visited):
            if current == target:
                return True
            visited.add(current)
            for conn in self.connections:
                if conn.in_node == current and conn.out_node not in visited:
                    if has_path(conn.out_node, target, visited):
                        return True
            return False

        if has_path(out_node, in_node, set()):
            print("Adding this connection would create a cycle. Mutation aborted.")
            return

        if weight is None:
            weight = random.uniform(-1.0, 1.0)
        new_conn = Connection(in_node=in_node, out_node=out_node, weight=weight)
        self.connections.append(new_conn)
        self.rebuild_graph()

    def mutate_remove_connection(self, in_node: int, out_node: int):
        """
        Mutate the network by removing a connection from in_node to out_node.
        """
        original_len = len(self.connections)
        self.connections = [
            conn
            for conn in self.connections
            if not (conn.in_node == in_node and conn.out_node == out_node)
        ]
        if len(self.connections) == original_len:
            print("No such connection found to remove.")
        else:
            self.rebuild_graph()

    def mutate_remove_node(self, node_id: int):
        """
        Mutate the network by removing the node with the given node_id.
        All connections with this node as either source or target are removed.
        """
        if node_id not in self.nodes:
            print("Node does not exist.")
            return

        # Remove the node.
        del self.nodes[node_id]
        # Remove all connections associated with this node.
        self.connections = [
            conn
            for conn in self.connections
            if conn.in_node != node_id and conn.out_node != node_id
        ]
        self.rebuild_graph()


# %%
if __name__ == "__main__":
    # Create input nodes.
    input_node0 = InputNode(0, label="x")
    input_node1 = InputNode(1, label="y")
    input_node2 = InputNode(2, label="d")

    # Create three hidden nodes, each with a different activation and aggregation.
    hidden_node1 = Node(
        activation="sigmoid", aggregation="sum", node_id=3, label=r"$\sigma$"
    )

    # Create an output node. For this example, we use identity activation and max aggregation.
    output_node = Node(activation="identity", aggregation="max", node_id=4)

    # List all nodes in the network.
    nodes = [
        input_node0,
        input_node1,
        input_node2,
        hidden_node1,
        output_node,
    ]

    # Create connections:
    # Hidden node 1 (node_id=2) receives inputs from both input nodes.
    connections = [
        Connection(in_node=0, out_node=3, weight=0.6),
        Connection(in_node=1, out_node=3, weight=0.4),
        Connection(in_node=2, out_node=3, weight=0.4),
        # Hidden node 2 (node_id=3) receives inputs from both input nodes with different weights.
        Connection(in_node=3, out_node=4, weight=0.8),
    ]

    # Build the CPPN network.
    cppn_net = CPPN(nodes, connections)

    # Prepare example input data.
    # Let's assume a batch size of 100. Each input should be of shape (100, 1).
    res = 50
    x_coords = jnp.linspace(-5, 5, res)
    y_coords = jnp.linspace(-5, 5, res)

    XX, YY = jnp.meshgrid(x_coords, y_coords)
    DD = jnp.sqrt(XX**2 + YY**2)
    # For a second input, we can use a sine transformation.
    # y_coords = jnp.sin(x_coords)

    inputs = {0: XX, 1: YY, 2: DD}

    # Perform the forward pass.
    output = cppn_net(inputs)
    plot_output(x_coords, y_coords, output)
    visualize_cppn_network(cppn_net)

    # remove a node and re-plot
    # cppn_net.mutate_remove_node(3)
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    cppn_net.mutate_add_node()
    visualize_cppn_network(cppn_net)
    output = cppn_net(inputs)
    plot_output(x_coords, y_coords, output)

# %%
