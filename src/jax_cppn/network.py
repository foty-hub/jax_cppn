# %%
from dataclasses import dataclass
import jax.numpy as jnp
import random
from jax import jit
from jax_cppn.node import Node, InputNode, OutputNode
from jax_cppn.vis import visualize_cppn_graph, plot_output

# TODO: make permitted activations an input to the mutate function
# TODO: add crossover
# TODO: add neat style evolution with a fitness function
# TODO: test different dimensionality inputs
# TODO: add cppn_saving functionality

PERMITTED_ACTIVATIONS = [
    "gauss",
    "sin",
    "sigmoid",
    "tanh",
]


@dataclass(frozen=True)
class Connection:
    in_node: int
    out_node: int
    weight: float


@dataclass(frozen=True)
class FunctionalCPPN:
    nodes: dict[int, Node]  # Mapping from node_id to Node.
    connections: list[Connection]  # list of all connections.
    topo_order: tuple[int, ...]  # Topologically sorted order of node ids.
    incoming: dict[
        int, list[Connection]
    ]  # Mapping from node_id to its incoming connections.

    def __hash__(self):
        # Build a tuple from sorted node info and connections.
        # We assume Node has attributes 'activation', 'aggregation', 'node_id', and 'label'.
        nodes_tuple = tuple(
            sorted(
                (nid, node.activation, node.aggregation, node.node_id, node.label)
                for nid, node in self.nodes.items()
            )
        )
        connections_tuple = tuple(
            sorted(
                (conn.in_node, conn.out_node, conn.weight) for conn in self.connections
            )
        )
        return hash((nodes_tuple, connections_tuple, self.topo_order))


def build_incoming(
    nodes: dict[int, Node], connections: list[Connection]
) -> dict[int, list[Connection]]:
    incoming = {node_id: [] for node_id in nodes}
    for conn in connections:
        incoming[conn.out_node].append(conn)
    return incoming


def topological_sort(
    nodes: dict[int, Node], connections: list[Connection]
) -> tuple[int, ...]:
    in_degree = {node_id: 0 for node_id in nodes}
    for conn in connections:
        in_degree[conn.out_node] += 1
    queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
    topo_order = []
    while queue:
        current = queue.pop(0)
        topo_order.append(current)
        for conn in connections:
            if conn.in_node == current:
                in_degree[conn.out_node] -= 1
                if in_degree[conn.out_node] == 0:
                    queue.append(conn.out_node)
    if len(topo_order) != len(nodes):
        raise ValueError(
            "Graph has cycles or disconnected parts; topological sort failed."
        )
    return tuple(topo_order)


def build_cppn(nodes: list[Node], connections: list[Connection]) -> FunctionalCPPN:
    nodes_dict = {node.node_id: node for node in nodes}
    incoming = build_incoming(nodes_dict, connections)
    topo_order = topological_sort(nodes_dict, connections)
    return FunctionalCPPN(
        nodes=nodes_dict,
        connections=connections,
        topo_order=topo_order,
        incoming=incoming,
    )


def _forward_cppn(
    cppn: FunctionalCPPN, inputs: dict[str, jnp.array]
) -> dict[str, jnp.array]:
    """
    Run a forward pass through the network using a pre-allocated JAX array
    to store intermediate computed node outputs.

    Assumes that node outputs share the same shape.

    Parameters:
        - cppn: The CPPN network.
        - inputs: A dictionary mapping input node labels to their corresponding JAX arrays.

    Returns:
        A dictionary mapping each output node's label to its computed output.
    """
    # Determine the shape of one input array.
    output_shape = next(iter(inputs.values())).shape
    # Total number of nodes: assume max id + 1.
    n_nodes = max(cppn.nodes.keys()) + 1

    # Pre-allocate a computed array of shape (n_nodes, *output_shape)
    computed = jnp.zeros((n_nodes, *output_shape))

    # Set the values for input nodes by matching labels.
    for node_id, node in cppn.nodes.items():
        if isinstance(node, InputNode):
            if node.label in inputs:
                computed = computed.at[node_id].set(inputs[node.label])
            else:
                raise ValueError(
                    f"Missing input for input node with label '{node.label}'."
                )

    # Process nodes in the pre-computed topological order.
    for node_id in cppn.topo_order:
        # Skip input nodes (they are already set).
        if isinstance(cppn.nodes[node_id], InputNode):
            continue
        # Compute the weighted inputs from all incoming connections.
        weighted_inputs = [
            conn.weight * computed[conn.in_node] for conn in cppn.incoming[node_id]
        ]
        # Stack along axis 1 so that the connections dimension is in the middle.
        stacked = jnp.stack(weighted_inputs, axis=1)
        # The aggregation function (e.g. sum) should sum along axis 1.
        aggregated = cppn.nodes[node_id].aggregation(stacked)
        activated = cppn.nodes[node_id].activation(aggregated)
        computed = computed.at[node_id].set(activated)

    # Gather computed outputs for all nodes that are instances of OutputNode.
    outputs = {}
    for node_id, node in cppn.nodes.items():
        if isinstance(node, OutputNode):
            outputs[node.label] = computed[node_id]

    return outputs
    # return computed[cppn.topo_order[-1]]


# Mark the network argument as static.
forward_cppn = jit(_forward_cppn, static_argnums=(0,))

# --- Mutation Operators (Functional Style) ---


def mutate_add_node(
    cppn: FunctionalCPPN, connection_index: int | None = None
) -> FunctionalCPPN:
    """
    Split an existing connection by adding a new node.
    """
    if not cppn.connections:
        print("No connections available to split.")
        return cppn

    connections = list(cppn.connections)
    if connection_index is None:
        connection_index = random.randrange(len(connections))
    old_conn = connections.pop(connection_index)

    # Create a new node with a new unique node id.
    new_node_id = max(cppn.nodes.keys()) + 1
    new_activation = random.choice(PERMITTED_ACTIVATIONS)
    label = r"$\sigma$" if new_activation == "sigmoid" else None
    new_node = Node(
        activation=new_activation, aggregation="sum", node_id=new_node_id, label=label
    )

    new_nodes = dict(cppn.nodes)
    new_nodes[new_node_id] = new_node

    # Create two new connections: one from the old in_node to the new node,
    # and one from the new node to the old out_node (carrying the old weight).
    conn1 = Connection(in_node=old_conn.in_node, out_node=new_node_id, weight=1.0)
    conn2 = Connection(
        in_node=new_node_id, out_node=old_conn.out_node, weight=old_conn.weight
    )
    new_connections = connections + [conn1, conn2]

    new_incoming = build_incoming(new_nodes, new_connections)
    new_topo = topological_sort(new_nodes, new_connections)
    return FunctionalCPPN(
        nodes=new_nodes,
        connections=new_connections,
        topo_order=new_topo,
        incoming=new_incoming,
    )


def mutate_add_connection(
    cppn: FunctionalCPPN, in_node: int, out_node: int, weight: float | None = None
) -> FunctionalCPPN:
    """
    Add a new connection from in_node to out_node if it doesn't already exist
    and doesn't create a cycle.
    """
    if in_node not in cppn.nodes or out_node not in cppn.nodes:
        raise ValueError("Invalid node ids provided for connection.")

    for conn in cppn.connections:
        if conn.in_node == in_node and conn.out_node == out_node:
            # print("Connection already exists.")
            return cppn

    # Simple DFS to check for a cycle.
    def has_path(current, target, visited):
        if current == target:
            return True
        visited.add(current)
        for conn in cppn.connections:
            if conn.in_node == current and conn.out_node not in visited:
                if has_path(conn.out_node, target, visited):
                    return True
        return False

    if has_path(out_node, in_node, set()):
        # print("Adding this connection would create a cycle. Mutation aborted.")
        return cppn

    if weight is None:
        weight = random.uniform(-1.0, 1.0)
    new_conn = Connection(in_node=in_node, out_node=out_node, weight=weight)
    new_connections = cppn.connections + [new_conn]
    new_incoming = build_incoming(cppn.nodes, new_connections)
    new_topo = topological_sort(cppn.nodes, new_connections)
    return FunctionalCPPN(
        nodes=cppn.nodes,
        connections=new_connections,
        topo_order=new_topo,
        incoming=new_incoming,
    )


def validate_cppn(nodes: dict[int, Node], connections: list[Connection]) -> bool:
    """
    Validate the CPPN network by ensuring:
      - Input nodes have at least one outgoing connection.
      - Output nodes have at least one incoming connection.
      - Hidden nodes are involved in at least one connection.
      - All nodes are reachable from at least one input node.
      - Every node can eventually feed into an output node.
    """
    # Basic connectivity: ensure every node participates in a connection.
    for nid, node in nodes.items():
        if isinstance(node, InputNode):
            if not any(conn.in_node == nid for conn in connections):
                return False
        elif isinstance(node, OutputNode):
            if not any(conn.out_node == nid for conn in connections):
                return False
        else:
            if not any(
                conn.in_node == nid or conn.out_node == nid for conn in connections
            ):
                return False

    # Verify that every node is reachable from some input node.
    input_ids = [nid for nid, node in nodes.items() if isinstance(node, InputNode)]
    if not input_ids:
        return False

    reachable_from_input = set()
    for nid in input_ids:
        stack = [nid]
        while stack:
            current = stack.pop()
            if current not in reachable_from_input:
                reachable_from_input.add(current)
                for conn in connections:
                    if (
                        conn.in_node == current
                        and conn.out_node not in reachable_from_input
                    ):
                        stack.append(conn.out_node)
    if set(nodes.keys()) != reachable_from_input:
        return False

    # Ensure there is at least one output node.
    output_ids = [nid for nid, node in nodes.items() if isinstance(node, OutputNode)]
    if not output_ids:
        return False

    # Check that every node can reach an output node using reverse DFS.
    reverse_adj = {nid: [] for nid in nodes}
    for conn in connections:
        reverse_adj[conn.out_node].append(conn.in_node)

    can_feed_into_output = set()
    stack = list(output_ids)
    while stack:
        current = stack.pop()
        if current not in can_feed_into_output:
            can_feed_into_output.add(current)
            for pred in reverse_adj[current]:
                if pred not in can_feed_into_output:
                    stack.append(pred)
    if set(nodes.keys()) != can_feed_into_output:
        return False

    return True


def mutate_remove_connection(
    cppn: FunctionalCPPN, in_node: int, out_node: int
) -> FunctionalCPPN:
    """
    Remove a connection from in_node to out_node.
    The mutation is aborted if removing the connection would
    leave any node isolated or disconnect the graph.
    """
    new_connections = [
        conn
        for conn in cppn.connections
        if not (conn.in_node == in_node and conn.out_node == out_node)
    ]
    if len(new_connections) == len(cppn.connections):
        # print("No such connection found to remove.")
        return cppn

    if not validate_cppn(cppn.nodes, new_connections):
        # print(
        #     "Mutation aborted: removing connection would result in an invalid network."
        # )
        return cppn

    try:
        new_topo = topological_sort(cppn.nodes, new_connections)
    except ValueError:
        # print("Mutation aborted due to topological sort failure:", e)
        return cppn

    new_incoming = build_incoming(cppn.nodes, new_connections)
    return FunctionalCPPN(
        nodes=cppn.nodes,
        connections=new_connections,
        topo_order=new_topo,
        incoming=new_incoming,
    )


def mutate_remove_node(cppn: FunctionalCPPN, node_id: int) -> FunctionalCPPN:
    """
    Remove a node (and all its associated connections) from the network.
    The mutation is aborted if:
      - The node does not exist,
      - The node is an input or output node (which we disallow),
      - Removal would leave any remaining node isolated, or
      - Removal disconnects the network.
    """
    if node_id not in cppn.nodes:
        # print("Node does not exist.")
        return cppn

    if isinstance(cppn.nodes[node_id], (InputNode, OutputNode)):
        # print("Cannot remove input or output node.")
        return cppn

    new_nodes = {nid: n for nid, n in cppn.nodes.items() if nid != node_id}
    new_connections = [
        conn
        for conn in cppn.connections
        if conn.in_node != node_id and conn.out_node != node_id
    ]

    if not validate_cppn(new_nodes, new_connections):
        # print("Mutation aborted: removing node would result in an invalid network.")
        return cppn

    try:
        new_topo = topological_sort(new_nodes, new_connections)
    except ValueError:
        # print("Mutation aborted due to topological sort failure:", e)
        return cppn

    new_incoming = build_incoming(new_nodes, new_connections)
    return FunctionalCPPN(
        nodes=new_nodes,
        connections=new_connections,
        topo_order=new_topo,
        incoming=new_incoming,
    )


def mutate(
    cppn: FunctionalCPPN,
    mutation_probs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> FunctionalCPPN:
    """
    Randomly mutates the given CPPN using one of the 4 mutation operators,
    chosen according to the provided relative probabilities.

    mutation_probs: A tuple of 4 floats corresponding to the relative weights for:
        (mutate_add_node, mutate_add_connection, mutate_remove_connection, mutate_remove_node).
    """
    total = sum(mutation_probs)
    normalized_probs = [p / total for p in mutation_probs]
    # Map: 0 -> add node, 1 -> add connection, 2 -> remove connection, 3 -> remove node.
    mutation_choice = random.choices([0, 1, 2, 3], weights=normalized_probs, k=1)[0]

    if mutation_choice == 0:
        return mutate_add_node(cppn)
    elif mutation_choice == 1:
        node_ids = list(cppn.nodes.keys())
        if len(node_ids) < 2:
            # print("Not enough nodes to add a connection.")
            return cppn

        # Build a list of valid (in_node, out_node) pairs.
        allowed_pairs = []
        for in_node in node_ids:
            for out_node in node_ids:
                if in_node == out_node:
                    continue
                # Do not allow connecting two input nodes.
                if isinstance(cppn.nodes[in_node], InputNode) and isinstance(
                    cppn.nodes[out_node], InputNode
                ):
                    continue
                # Do not allow connecting two output nodes.
                if isinstance(cppn.nodes[in_node], OutputNode) and isinstance(
                    cppn.nodes[out_node], OutputNode
                ):
                    continue
                allowed_pairs.append((in_node, out_node))

        if not allowed_pairs:
            # print("No valid node pairs available for connection mutation.")
            return cppn

        chosen_pair = random.choice(allowed_pairs)
        return mutate_add_connection(cppn, chosen_pair[0], chosen_pair[1])
    elif mutation_choice == 2:
        if not cppn.connections:
            # print("No connections to remove.")
            return cppn
        conn = random.choice(cppn.connections)
        return mutate_remove_connection(cppn, conn.in_node, conn.out_node)
    elif mutation_choice == 3:
        node_ids = list(cppn.nodes.keys())
        if not node_ids:
            # print("No nodes to remove.")
            return cppn
        node_id = random.choice(node_ids)
        return mutate_remove_node(cppn, node_id)


def init_cppn(
    input_node_labels: list[str],
    output_node_labels: list[str],
    hidden_activation: str = "sigmoid",
    hidden_aggregation: str = "sum",
):
    "Builds a simple CPPN with inputs, outputs, and one hidden node to begin mutating"
    nodes = []
    connections = []

    # add the hidden node for everything else to connect to
    hidden = Node(
        activation=hidden_activation,
        aggregation=hidden_aggregation,
        node_id=0,
    )
    nodes.append(hidden)

    # Add input nodes
    next_node_id = 1
    for label in input_node_labels:
        node_id = next_node_id
        nodes.append(InputNode(node_id=node_id, label=label))
        connections.append(Connection(in_node=node_id, out_node=0, weight=1.0))
        next_node_id += 1

    # Add output nodes
    for label in output_node_labels:
        node_id = next_node_id
        nodes.append(OutputNode(node_id=node_id, label=label))
        connections.append(Connection(in_node=0, out_node=node_id, weight=1.0))
        next_node_id += 1

    return build_cppn(nodes, connections)


# --- Example usage ---

if __name__ == "__main__":
    # initialise the coordinate space
    res = 256
    x_coords = jnp.linspace(-1, 1, res)
    y_coords = jnp.linspace(-1, 1, res)
    XX, YY = jnp.meshgrid(x_coords, y_coords)
    DD = jnp.sqrt(XX**2 + YY**2)
    inputs = {"x": XX, "y": YY, "d": DD}

    # Start a network with one node
    cppn_net = init_cppn(["x", "y", "d"], ["out"])

    # (add_node, add_connection, remove_node, remove_connection)
    mutation_probs = (0.15, 0.6, 0.1, 0.2)
    for _ in range(50):
        cppn_net = mutate(cppn_net, mutation_probs)

    output = forward_cppn(cppn_net, inputs)
    plot_output(x_coords, y_coords, output["out"])
    visualize_cppn_graph(cppn_net)

    # "zoom out" by increasing the extent of XX, YY
    # res = 1024
    # x_coords = jnp.linspace(-5, 5, res)
    # y_coords = jnp.linspace(-5, 5, res)
    # XX, YY = jnp.meshgrid(x_coords, y_coords)
    # DD = jnp.sqrt(XX**2 + YY**2)
    # inputs = {"x": XX, "y": YY, "d": DD}
    # output = forward_cppn(cppn_net, inputs)
    # plot_output(x_coords, y_coords, output["out"])
# %%
