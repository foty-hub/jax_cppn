# %%
from dataclasses import dataclass
import jax.numpy as jnp
import random
from jax import jit
from jax_cppn.node import Node, InputNode, OutputNode
from jax_cppn.vis import visualize_cppn_network, plot_output

# TODO: add crossover
# TODO: add neat style evolution with a fitness function
# TODO: test different dimensionality inputs

PERMITTED_MUTATIONS = [
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


def forward_cppn(cppn: FunctionalCPPN, inputs: dict[int, jnp.array]) -> jnp.array:
    """
    Run a forward pass through the network using a pre-allocated JAX array
    to store intermediate computed node outputs.

    Assumes that node IDs are integers starting at 0 and that all node outputs share the same shape.
    """
    # Determine the shape of one input array.
    output_shape = next(iter(inputs.values())).shape
    # Total number of nodes: assume max id + 1.
    n_nodes = max(cppn.nodes.keys()) + 1

    # Pre-allocate a computed array of shape (n_nodes, H, W)
    computed = jnp.zeros((n_nodes, *output_shape))

    # Set the values for input nodes.
    for node_id, value in inputs.items():
        computed = computed.at[node_id].set(value)

    # Process nodes in the pre-computed topological order.
    for node_id in cppn.topo_order:
        # Skip input nodes (they are already set).
        if node_id in inputs:
            continue
        # Compute the weighted inputs from all incoming connections.
        weighted_inputs = [
            conn.weight * computed[conn.in_node] for conn in cppn.incoming[node_id]
        ]
        # Stack along axis 1 so that the connections dimension is in the middle.
        stacked = jnp.stack(
            weighted_inputs, axis=1
        )  # Shape: (H, num_connections, W) if each computed[...] is (H, W)
        # The aggregation function (e.g. sum) should sum along axis 1.
        aggregated = cppn.nodes[node_id].aggregation(stacked)
        activated = cppn.nodes[node_id].activation(aggregated)
        computed = computed.at[node_id].set(activated)

    # Return the output from the final node (last in topological order).
    return computed[cppn.topo_order[-1]]


# Mark the network argument as static.
jit_forward_cppn = jit(forward_cppn, static_argnums=(0,))

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
    new_activation = random.choice(PERMITTED_MUTATIONS)
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
    Validate that:
      - Input nodes have at least one outgoing connection.
      - Output nodes have at least one incoming connection.
      - Other (hidden) nodes are involved in at least one connection.
      - All nodes are reachable from at least one input node.
      - Each input node can reach at least one output node.
      - Every node can reach at least one output node (no dead-end hidden nodes).
    """

    # --- Existing checks ---
    for nid, node in nodes.items():
        if isinstance(node, InputNode):
            # Input nodes must have at least one outgoing connection.
            if not any(conn.in_node == nid for conn in connections):
                # print(f"Input node {nid} would have no outgoing connections.")
                return False
        elif isinstance(node, OutputNode):
            # Output nodes must have at least one incoming connection.
            if not any(conn.out_node == nid for conn in connections):
                # print(f"Output node {nid} would have no incoming connections.")
                return False
        else:
            # Other nodes must appear in at least one connection.
            if not any(
                conn.in_node == nid or conn.out_node == nid for conn in connections
            ):
                # print(f"Node {nid} would become isolated.")
                return False

    # Check that all nodes are reachable from some input node (no fully disconnected subgraphs).
    input_ids = [nid for nid, node in nodes.items() if isinstance(node, InputNode)]
    if not input_ids:
        # print("No input nodes present in the graph.")
        return False

    reachable_from_any_input = set()
    for nid in input_ids:
        stack = [nid]
        while stack:
            current = stack.pop()
            if current not in reachable_from_any_input:
                reachable_from_any_input.add(current)
                for conn in connections:
                    if (
                        conn.in_node == current
                        and conn.out_node not in reachable_from_any_input
                    ):
                        stack.append(conn.out_node)
    if set(nodes.keys()) != reachable_from_any_input:
        # print("Graph is disconnected: some nodes are unreachable from input nodes.")
        return False

    # Ensure at least one output node exists.
    output_ids = [nid for nid, node in nodes.items() if isinstance(node, OutputNode)]
    if not output_ids:
        # print("No output nodes present in the graph.")
        return False

    # Each input node can reach at least one output node.
    for in_id in input_ids:
        stack = [in_id]
        reachable_from_in = set()
        while stack:
            current = stack.pop()
            if current not in reachable_from_in:
                reachable_from_in.add(current)
                for conn in connections:
                    if (
                        conn.in_node == current
                        and conn.out_node not in reachable_from_in
                    ):
                        stack.append(conn.out_node)
        # If none of the output nodes is reachable from this input node, fail.
        if not any(out_id in reachable_from_in for out_id in output_ids):
            # print(
            #     f"Input node {in_id} cannot reach any output node. "
            #     "Network is validly connected but input is effectively unused."
            # )
            return False

    # Check that every node can reach an output node ---
    # Build a reverse adjacency list: for each node, which nodes feed into it?
    reverse_adj = {nid: [] for nid in nodes}
    for conn in connections:
        reverse_adj[conn.out_node].append(conn.in_node)

    # BFS/DFS from each output node in the reverse graph:
    can_feed_into_output = set()
    stack = list(output_ids)
    while stack:
        current = stack.pop()
        if current not in can_feed_into_output:
            can_feed_into_output.add(current)
            for pred in reverse_adj[current]:
                if pred not in can_feed_into_output:
                    stack.append(pred)

    # If any node is not in can_feed_into_output, it cannot reach any output node.
    all_nodes = set(nodes.keys())
    dead_end_nodes = all_nodes - can_feed_into_output
    if dead_end_nodes:
        # print(
        #     f"These node(s) cannot feed into any output: {dead_end_nodes}. "
        #     "No path from them to an output node."
        # )
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
    except ValueError as e:
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


# --- Example usage ---

if __name__ == "__main__":
    input_node0 = InputNode(0, label="x")
    input_node1 = InputNode(1, label="y")
    input_node2 = InputNode(2, label="d")
    hidden_node1 = Node(
        activation="sigmoid", aggregation="sum", node_id=3, label=r"$\sigma$"
    )
    output_node = OutputNode(4, label="out")
    nodes = [input_node0, input_node1, input_node2, hidden_node1, output_node]
    connections = [
        Connection(in_node=0, out_node=3, weight=0.6),
        Connection(in_node=1, out_node=3, weight=0.4),
        Connection(in_node=2, out_node=3, weight=0.4),
        Connection(in_node=3, out_node=4, weight=0.8),
    ]
    cppn_net = build_cppn(nodes, connections)
    print("Functional network structure:")
    print(cppn_net)

    for _ in range(100):
        cppn_net = mutate(cppn_net)

    res = 256
    x_coords = jnp.linspace(-1, 1, res)
    y_coords = jnp.linspace(-1, 1, res)
    XX, YY = jnp.meshgrid(x_coords, y_coords)
    DD = jnp.sqrt(XX**2 + YY**2)
    inputs = {0: XX, 1: YY, 2: DD}

    output = jit_forward_cppn(cppn_net, inputs)
    plot_output(x_coords, y_coords, output)
    visualize_cppn_network(cppn_net)
# %%
