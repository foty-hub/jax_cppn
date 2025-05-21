# %%
"""Core module for defining and operating on Compositional Pattern Producing Networks (CPPNs).

This module provides the `FunctionalCPPN` dataclass to represent the static structure
of a CPPN, including its nodes, connections, and topological order. It offers
functions to build, validate, and perform forward propagation through these networks.
Additionally, a suite of mutation operators is included to enable evolutionary
algorithms or other generative processes to modify and explore different CPPN architectures.

The CPPNs defined here are "functional" in the sense that their structure is fixed
at the time of creation (or mutation, which creates a new instance), and the
forward pass is a pure function. JAX is used for JIT compilation of the forward
pass for performance.

Key components:
    - `Connection`: Represents a weighted connection between two nodes.
    - `FunctionalCPPN`: An immutable representation of the CPPN's graph structure.
    - `build_cppn`: Constructs a `FunctionalCPPN` from nodes and connections.
    - `forward_cppn`: Performs a forward pass through the network.
    - Mutation functions (`mutate_add_node`, `mutate_add_connection`, etc.):
      Functions that take a `FunctionalCPPN` and return a new, modified instance.
    - `init_cppn`: Initializes a basic CPPN structure.
"""
from dataclasses import dataclass
import jax.numpy as jnp
import random
from jax import jit
from jax_cppn.node import Node, InputNode, OutputNode
from jax_cppn.vis import visualize_cppn_graph, plot_output
from typing import Dict, List, Tuple, Optional # For type hinting

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
    """Represents a directed, weighted connection between two nodes in a CPPN.

    This is an immutable dataclass.

    Attributes:
        in_node (int): The ID of the node from which the connection originates.
        out_node (int): The ID of the node to which the connection leads.
        weight (float): The weight associated with this connection.
    """
    in_node: int
    out_node: int
    weight: float


@dataclass(frozen=True)
class FunctionalCPPN:
    """An immutable representation of a Compositional Pattern Producing Network (CPPN).

    This dataclass stores the complete structure of a CPPN, including all its nodes,
    the connections between them, a topologically sorted order for efficient
    computation, and a mapping of nodes to their incoming connections.
    Being a frozen dataclass, instances of `FunctionalCPPN` are immutable.
    Functions that "modify" a CPPN, such as mutation operators, actually return
    a new `FunctionalCPPN` instance.

    Attributes:
        nodes (Dict[int, Node]): A dictionary mapping unique node IDs (int) to
            their corresponding `Node` objects.
        connections (List[Connection]): A list of all `Connection` objects
            that define the links and weights between nodes in the network.
        topo_order (Tuple[int, ...]): A tuple of node IDs, topologically sorted.
            This order ensures that when traversing the network, a node is
            processed only after all its input nodes have been processed.
            Essential for correct feed-forward computation in a directed acyclic graph.
        incoming (Dict[int, List[Connection]]): A dictionary mapping each node ID
            to a list of `Connection` objects that terminate at that node.
            This facilitates quick lookup of all inputs to a given node during
            the forward pass.
    """
    nodes: Dict[int, Node]  # Mapping from node_id to Node.
    connections: List[Connection]  # list of all connections.
    topo_order: Tuple[int, ...]  # Topologically sorted order of node ids.
    incoming: Dict[
        int, List[Connection]
    ]  # Mapping from node_id to its incoming connections.

    def __hash__(self):
        """Computes a hash for the CPPN based on its structure.

        This allows `FunctionalCPPN` instances to be used in sets or as dictionary keys,
        provided their `Node` objects are also hashable or consistently represented.
        The hash is derived from a sorted tuple representation of nodes, connections,
        and the topological order, ensuring that structurally identical CPPNs
        have the same hash.

        Returns:
            int: The hash value for this `FunctionalCPPN` instance.
        """
        # Build a tuple from sorted node info and connections.
        # We assume Node has attributes 'activation', 'aggregation', 'node_id', and 'label'.
        # For hashing, we use act_str and agg_str as functions are not directly hashable.
        nodes_tuple = tuple(
            sorted(
                (nid, node.act_str, node.agg_str, node.node_id, node.label)
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
    nodes: Dict[int, Node], connections: List[Connection]
) -> Dict[int, List[Connection]]:
    """Constructs a mapping of node IDs to their list of incoming connections.

    This helper function is used during the CPPN building process (`build_cppn`).
    The resulting dictionary is crucial for efficiently accessing all input
    signals to a specific node during the network's forward pass.

    Args:
        nodes: A dictionary mapping node IDs to `Node` objects.
        connections: A list of `Connection` objects representing the network's graph.

    Returns:
        A dictionary where keys are node IDs and values are lists of `Connection`
        objects that have the key node as their `out_node`.
    """
    incoming = {node_id: [] for node_id in nodes}
    for conn in connections:
        incoming[conn.out_node].append(conn)
    return incoming


def topological_sort(
    nodes: Dict[int, Node], connections: List[Connection]
) -> Tuple[int, ...]:
    """Performs a topological sort of the CPPN nodes.

    This function implements Kahn's algorithm for topological sorting. The sort
    is essential for processing nodes in a directed acyclic graph (DAG) in an order
    that ensures all inputs to a node are processed before the node itself.
    This is fundamental for the correct feed-forward evaluation of the CPPN.

    Args:
        nodes: A dictionary mapping node IDs to `Node` objects.
        connections: A list of `Connection` objects.

    Returns:
        A tuple of node IDs in topologically sorted order.

    Raises:
        ValueError: If the graph contains a cycle (is not a DAG) or has
            disconnected parts that prevent a full topological sort of all nodes.
    """
    in_degree = {node_id: 0 for node_id in nodes}
    # Adjacency list: maps a node to a list of nodes it connects to.
    adj = {node_id: [] for node_id in nodes}

    for conn in connections:
        in_degree[conn.out_node] += 1
        adj[conn.in_node].append(conn.out_node)

    queue = [node_id for node_id, deg in in_degree.items() if deg == 0]
    topo_order = []

    while queue:
        current = queue.pop(0)
        topo_order.append(current)
        # For each neighbor of the current node
        for neighbor_node_id in adj[current]:
            in_degree[neighbor_node_id] -= 1
            if in_degree[neighbor_node_id] == 0:
                queue.append(neighbor_node_id)

    if len(topo_order) != len(nodes):
        # Identify problematic nodes for better error message
        missing_nodes = set(nodes.keys()) - set(topo_order)
        # Check for cycles by seeing if any missing node still has in-degree > 0
        # or if they were simply not reachable from the initial queue (disconnected graph part)
        # This part of error reporting could be more sophisticated if needed.
        raise ValueError(
            f"Graph has cycles or disconnected parts; topological sort failed. "
            f"Processed {len(topo_order)} nodes out of {len(nodes)}. "
            f"Missing or problematic nodes might include: {missing_nodes}"
        )
    return tuple(topo_order)


def build_cppn(nodes: List[Node], connections: List[Connection]) -> FunctionalCPPN:
    """Constructs a `FunctionalCPPN` instance from a list of nodes and connections.

    This function orchestrates the creation of a complete CPPN structure. It involves:
    1. Converting the list of nodes into a dictionary for quick lookup.
    2. Building the `incoming` connection map using `build_incoming`.
    3. Performing a `topological_sort` to establish the correct processing order.
    It then instantiates and returns a `FunctionalCPPN` object. As `FunctionalCPPN`
    is immutable, this function effectively creates a new, self-contained network representation.

    Args:
        nodes: A list of `Node` objects (including `InputNode` and `OutputNode` instances).
        connections: A list of `Connection` objects defining the network topology and weights.

    Returns:
        A new, immutable `FunctionalCPPN` instance.
    """
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
    cppn: FunctionalCPPN, inputs: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """Performs a forward pass through the CPPN to compute outputs.

    This is the core computational function of the network. It takes a `FunctionalCPPN`
    structure and a dictionary of input values (mapping input node labels to JAX arrays)
    and propagates these values through the network according to its topology,
    aggregation functions, and activation functions.

    The process involves:
    1. Initializing a data structure (`computed`) to store the output of each node.
       The size of this structure is determined by the maximum node ID and the shape
       of the input arrays. It's assumed all inputs (and thus all node outputs)
       will have the same shape (e.g., a 2D grid of coordinates).
    2. Assigning the provided `inputs` to their corresponding `InputNode` entries in `computed`.
    3. Iterating through the nodes in `topo_order`:
        a. For each node, gather the outputs of its prerequisite nodes (already computed)
           from the `computed` array, using the `incoming` connection map.
        b. Apply the connection weights to these gathered inputs.
        c. Aggregate these weighted inputs using the node's `aggregation` function.
        d. Apply the node's `activation` function to the aggregated result.
        e. Store this final output in the `computed` array for the current node.
    4. After processing all nodes, collect the values from `computed` that correspond
       to `OutputNode` instances and return them in a dictionary mapping output labels
       to their JAX array outputs.

    Args:
        cppn: The `FunctionalCPPN` instance representing the network structure.
        inputs: A dictionary mapping input node labels (str) to their corresponding
            JAX numpy arrays. These arrays typically represent spatial coordinates
            (e.g., x, y, distance from center) and must all have the same shape.
            Example: `{"x": XX, "y": YY}` where XX, YY are meshgrid arrays.
            (See `examples/rgb_cppn.py` for usage).

    Returns:
        A dictionary mapping output node labels (str) to the computed JAX numpy arrays.
        The shape of these output arrays will match the shape of the input arrays.

    Raises:
        ValueError: If an input is not provided for a defined `InputNode`.
    """
    # Determine the shape of one input array. Assumes all inputs have the same shape.
    if not inputs:
        # Handle case with no inputs, though typically CPPNs have inputs.
        # If there are output nodes that don't depend on inputs (e.g. constant output),
        # this might be valid, but usually, inputs are expected.
        # For now, let's assume if there are nodes, there should be inputs or specific handling.
        # If cppn.nodes is also empty, then an empty dict is fine.
        if not cppn.nodes:
            return {}
        # If there are nodes but no inputs, and input nodes exist, it's an issue.
        if any(isinstance(node, InputNode) for node in cppn.nodes.values()):
            raise ValueError("Inputs dictionary is empty, but the CPPN has InputNodes.")
        # If no input nodes, proceed, assuming output_shape can be (1,) or similar default.
        output_shape = (1,) # Default shape if no inputs to infer from.
    else:
        output_shape = next(iter(inputs.values())).shape

    # Total number of potential node slots: max id + 1.
    # This assumes node IDs are somewhat contiguous, starting from 0 or a small number.
    # If node IDs can be very large and sparse, this could be memory inefficient.
    # Consider a dictionary for `computed` if node IDs are sparse and large.
    n_nodes = max(cppn.nodes.keys()) + 1 if cppn.nodes else 0


    # Pre-allocate a computed array of shape (n_nodes, *output_shape)
    # Ensure output_shape is not empty, e.g. for scalar inputs/outputs
    if not output_shape: # Handle scalar case, e.g. shape ()
        computed_shape = (n_nodes,)
    else:
        computed_shape = (n_nodes, *output_shape)
    computed = jnp.zeros(computed_shape, dtype=jnp.float32) # Assuming float32

    # Set the values for input nodes by matching labels.
    for node_id, node in cppn.nodes.items():
        if isinstance(node, InputNode):
            if node.label and node.label in inputs:
                computed = computed.at[node_id].set(inputs[node.label])
            else:
                # This check is important: if an InputNode is defined in the CPPN,
                # it must receive a corresponding input value.
                raise ValueError(
                    f"Missing input for input node with label '{node.label}' (ID: {node_id})."
                )

    # Process nodes in the pre-computed topological order.
    for node_id in cppn.topo_order:
        node = cppn.nodes[node_id]
        # Skip input nodes (they are already set and have no incoming connections to process here).
        if isinstance(node, InputNode):
            continue

        # Get incoming connections for the current node
        incoming_conns = cppn.incoming.get(node_id, [])

        if not incoming_conns:
            # This node has no inputs (e.g., a bias node or an unconnected hidden/output node).
            # If it's an output node or a hidden node that's supposed to receive input,
            # this might indicate a structural issue, but mathematically, it means it receives no signal.
            # We can assume its input for aggregation is an empty list or a zero array of appropriate shape.
            # For simplicity, if aggregation needs an input, it might fail or produce a default.
            # Let's assume aggregation functions can handle an empty list of inputs if that's meaningful,
            # or we might need to pass a zero array.
            # For now, if a non-InputNode has no incoming connections, its `aggregated` value will
            # depend on how the aggregation function handles an empty stack.
            # Sum of empty is 0, product of empty is 1. This seems reasonable.
            # If using `jnp.stack([])`, it will raise an error.
            # So, if no incoming_conns, the 'aggregated' value should be a default,
            # often 0, for sum-like aggregations.
            # Let's define a default input for aggregation if no connections.
            # The shape should be (batch_size, 0) or similar for aggregation to work.
            # Or, handle it based on the aggregation function type.
            # For now, we'll assume aggregation functions can handle being called with an empty list,
            # or that valid CPPNs ensure non-input nodes have inputs.
            # A more robust way: if not weighted_inputs, aggregated = jnp.zeros(output_shape) or similar
            if node.aggregation.__name__ == "sum_aggregation":
                 aggregated_input_for_node = jnp.zeros(output_shape) # Sum of nothing is 0
            elif node.aggregation.__name__ == "product_aggregation":
                 aggregated_input_for_node = jnp.ones(output_shape) # Product of nothing is 1
            else:
                # For other aggregations, this might need specific handling or raise an error.
                # Or, rely on validate_cppn to prevent such nodes.
                # For now, let's assume it results in zeros.
                aggregated_input_for_node = jnp.zeros(output_shape)
        else:
            weighted_inputs = [
                conn.weight * computed[conn.in_node] for conn in incoming_conns
            ]
            # Stack along a new axis (axis=0 for list of (output_shape)) then transpose,
            # or stack along axis=1 if output_shape is (N,) making it (N, C)
            # If output_shape is (H, W), then weighted_inputs is a list of (H,W) arrays.
            # Stacking them creates (num_connections, H, W).
            # Aggregation expects (batch_dims..., num_connections_dim)
            # So, if output_shape is (X, Y), stack gives (C, X, Y).
            # We need to move C to the end: (X, Y, C) for aggregation over C.
            stacked = jnp.stack(weighted_inputs, axis=0) # Shape: (num_connections, *output_shape)
            # Transpose to put num_connections at the end for aggregation
            # axes_order = tuple(range(1, stacked.ndim)) + (0,) # e.g., (1, 2, 0) for 3D stack
            # stacked_transposed = jnp.transpose(stacked, axes=axes_order)
            # Aggregation functions in this project expect (N, C) and aggregate over C.
            # If inputs are 2D (e.g. images), output_shape = (H,W)
            # computed[conn.in_node] is (H,W). weighted_inputs is list of (H,W)
            # stacked is (C, H, W). We need to aggregate over C.
            # This requires aggregation functions to be flexible with input shapes or
            # reshape here. Assuming aggregation functions take (..., C) and reduce last axis.
            # The current aggregation functions (e.g. sum_aggregation) expect (N,C) and sum over axis 1.
            # This means the 'batch' dimension N here is our *output_shape.
            # If output_shape is (H,W), then N = H*W. We need to reshape.
            # Original input (e.g. XX) has shape (H,W).
            # computed[node_id] also has shape (H,W).
            # weighted_inputs is a list of C arrays, each of shape (H,W).
            # jnp.stack(weighted_inputs, axis=-1) would make it (H,W,C). This is good.
            # Then aggregation needs to work on the last axis.
            # Let's adjust stacking to be on the last axis.
            stacked_for_aggregation = jnp.stack(weighted_inputs, axis=-1) # Shape: (*output_shape, num_connections)
            aggregated_input_for_node = node.aggregation(stacked_for_aggregation)


        activated = node.activation(aggregated_input_for_node)
        computed = computed.at[node_id].set(activated)

    # Gather computed outputs for all nodes that are instances of OutputNode.
    outputs = {}
    for node_id, node in cppn.nodes.items():
        if isinstance(node, OutputNode) and node.label:
            outputs[node.label] = computed[node_id]

    return outputs


# JIT compile the forward pass function. `cppn` is static as its structure doesn't change per call.
forward_cppn = jit(_forward_cppn, static_argnums=(0,))
# Example for forward_cppn (conceptual, adapt from existing examples):
# >>> from jax_cppn.network import init_cppn, forward_cppn
# >>> import jax.numpy as jnp
# >>> # Ensure Node and other classes are defined if running standalone
# >>> cppn = init_cppn(input_node_labels=["x", "y"], output_node_labels=["out"]) # Simplified init
# >>> x_coords = jnp.linspace(-1, 1, 3)
# >>> y_coords = jnp.linspace(-1, 1, 3)
# >>> XX, YY = jnp.meshgrid(x_coords, y_coords)
# >>> inputs = {"x": XX, "y": YY}
# >>> outputs = forward_cppn(cppn, inputs) # JIT compilation happens on first call
# >>> print(outputs['out'].shape)
# (3, 3)

# --- Mutation Operators (Functional Style) ---


def mutate_add_node(
    cppn: FunctionalCPPN, connection_index: Optional[int] = None
) -> FunctionalCPPN:
    """Adds a new node by splitting an existing connection.

    A connection is chosen (randomly if `connection_index` is None). This
    connection (A -> B with weight W) is removed. A new node (N) is inserted.
    Two new connections are added: A -> N with weight 1.0, and N -> B with
    the original weight W. The new node N gets a randomly selected activation
    function from `PERMITTED_ACTIVATIONS` and a default "sum" aggregation.

    Args:
        cppn: The `FunctionalCPPN` to mutate.
        connection_index: Optional. The index of the connection in `cppn.connections`
            to split. If None, a random connection is chosen.

    Returns:
        A new `FunctionalCPPN` instance with the added node and modified
        connections. If no connections exist to split, it returns the original
        `cppn` unchanged and prints a message.
    """
    if not cppn.connections:
        # print("No connections available to split.") # Keep prints for debugging if desired
        return cppn

    connections = list(cppn.connections) # Make a mutable copy
    if connection_index is None:
        connection_index = random.randrange(len(connections))
    elif connection_index < 0 or connection_index >= len(connections):
        raise ValueError("connection_index is out of bounds.")

    old_conn = connections.pop(connection_index)

    # Create a new node with a new unique node id.
    new_node_id = max(cppn.nodes.keys()) + 1 if cppn.nodes else 0
    new_activation = random.choice(PERMITTED_ACTIVATIONS)
    # Simple label based on activation for visualization, can be None or more complex.
    label_map = {"sigmoid": r"$\sigma$", "tanh": "tanh", "gauss": "gauss", "sin": "sin"}
    label = label_map.get(new_activation)

    new_node = Node(
        activation=new_activation, aggregation="sum", node_id=new_node_id, label=label
    )

    new_nodes_dict = dict(cppn.nodes)
    new_nodes_dict[new_node_id] = new_node

    # Create two new connections
    conn1 = Connection(in_node=old_conn.in_node, out_node=new_node_id, weight=1.0)
    conn2 = Connection(
        in_node=new_node_id, out_node=old_conn.out_node, weight=old_conn.weight
    )
    new_connections_list = connections + [conn1, conn2]

    # Rebuild necessary FunctionalCPPN components
    return build_cppn(list(new_nodes_dict.values()), new_connections_list)


def mutate_add_connection(
    cppn: FunctionalCPPN, in_node_id: int, out_node_id: int, weight: Optional[float] = None
) -> FunctionalCPPN:
    """Adds a new connection between two existing nodes.

    A new connection is added from `in_node_id` to `out_node_id`.
    The mutation is aborted (original `cppn` returned) if:
    - The specified `in_node_id` or `out_node_id` does not exist.
    - A connection already exists between these two nodes in the same direction.
    - Adding the connection would create a cycle in the graph, which is checked
      by attempting a topological sort after adding the connection.

    Args:
        cppn: The `FunctionalCPPN` to mutate.
        in_node_id: The ID of the node where the connection starts.
        out_node_id: The ID of the node where the connection ends.
        weight: Optional. The weight for the new connection. If None, a random
            weight between -1.0 and 1.0 is chosen.

    Returns:
        A new `FunctionalCPPN` instance with the added connection. If the
        mutation is aborted for any reason, the original `cppn` is returned.

    Raises:
        ValueError: If `in_node_id` or `out_node_id` are not valid node IDs in the CPPN.
    """
    if in_node_id not in cppn.nodes or out_node_id not in cppn.nodes:
        raise ValueError("Invalid node IDs provided for connection.")

    # Prevent self-loops and connections to input nodes or from output nodes (common constraints)
    if in_node_id == out_node_id:
        # print("Cannot add self-loop connection.")
        return cppn
    if isinstance(cppn.nodes[out_node_id], InputNode):
        # print("Cannot add connection to an InputNode.")
        return cppn
    if isinstance(cppn.nodes[in_node_id], OutputNode):
        # print("Cannot add connection from an OutputNode.")
        return cppn


    for conn in cppn.connections:
        if conn.in_node == in_node_id and conn.out_node == out_node_id:
            # print("Connection already exists.")
            return cppn # Connection already exists

    # Tentatively add the new connection
    if weight is None:
        weight = random.uniform(-1.0, 1.0)
    new_conn = Connection(in_node=in_node_id, out_node=out_node_id, weight=weight)
    potential_new_connections = cppn.connections + [new_conn]

    # Check for cycles by attempting a topological sort.
    # This is a robust way to ensure the graph remains a DAG.
    try:
        # build_cppn will perform topological sort and raise ValueError if cycle
        return build_cppn(list(cppn.nodes.values()), potential_new_connections)
    except ValueError: # Typically from topological_sort failing due to a cycle
        # print("Adding this connection would create a cycle. Mutation aborted.")
        return cppn


def validate_cppn(nodes: Dict[int, Node], connections: List[Connection]) -> bool:
    """Validates the structural integrity of a CPPN.

    Performs several checks to ensure the CPPN is well-formed:
    1.  Input Nodes: Each `InputNode` must have at least one outgoing connection.
    2.  Output Nodes: Each `OutputNode` must have at least one incoming connection.
    3.  Hidden Nodes: Regular `Node` instances (hidden nodes) must be involved in
        at least one connection (either incoming or outgoing).
    4.  Reachability from Inputs: All nodes in the graph must be reachable from at
        least one `InputNode`. This prevents disconnected subgraphs that don't
        process any input.
    5.  Path to Outputs: All nodes must have a path leading to at least one
        `OutputNode`. This prevents parts of the network whose computations
        never contribute to an output.
    6.  Existence of Inputs/Outputs: The graph must contain at least one `InputNode`
        and at least one `OutputNode`.

    Args:
        nodes: A dictionary of node IDs to `Node` objects.
        connections: A list of `Connection` objects.

    Returns:
        True if the CPPN passes all validation checks, False otherwise.
    """
    if not nodes: return False
    node_ids = set(nodes.keys())

    input_node_ids = {nid for nid, n in nodes.items() if isinstance(n, InputNode)}
    output_node_ids = {nid for nid, n in nodes.items() if isinstance(n, OutputNode)}
    hidden_node_ids = node_ids - input_node_ids - output_node_ids

    if not input_node_ids or not output_node_ids:
        return False # Must have at least one input and one output node.

    # Check basic connectivity for each node type
    for nid in input_node_ids:
        if not any(conn.in_node == nid for conn in connections): return False
    for nid in output_node_ids:
        if not any(conn.out_node == nid for conn in connections): return False
    for nid in hidden_node_ids:
        if not any(conn.in_node == nid or conn.out_node == nid for conn in connections): return False

    # Build adjacency lists for graph traversals
    adj = {nid: [] for nid in node_ids}
    rev_adj = {nid: [] for nid in node_ids}
    for conn in connections:
        if conn.in_node in adj and conn.out_node in adj: # Ensure conn nodes are in current nodes set
            adj[conn.in_node].append(conn.out_node)
            rev_adj[conn.out_node].append(conn.in_node)

    # 4. Reachability from Inputs: All nodes must be reachable from an InputNode.
    q = list(input_node_ids)
    reachable_from_inputs = set(input_node_ids)
    head = 0
    while head < len(q):
        curr = q[head]
        head += 1
        for neighbor in adj.get(curr, []):
            if neighbor not in reachable_from_inputs:
                reachable_from_inputs.add(neighbor)
                q.append(neighbor)
    if reachable_from_inputs != node_ids: return False

    # 5. Path to Outputs: All nodes must have a path to an OutputNode.
    q = list(output_node_ids)
    can_reach_outputs = set(output_node_ids)
    head = 0
    while head < len(q):
        curr = q[head]
        head += 1
        for neighbor in rev_adj.get(curr, []): # Traverse backwards from outputs
            if neighbor not in can_reach_outputs:
                can_reach_outputs.add(neighbor)
                q.append(neighbor)
    if can_reach_outputs != node_ids: return False

    return True


def mutate_remove_connection(
    cppn: FunctionalCPPN, in_node_id: int, out_node_id: int
) -> FunctionalCPPN:
    """Removes a specific connection between `in_node_id` and `out_node_id`.

    The mutation is aborted (original `cppn` returned) if:
    - The specified connection does not exist.
    - Removing the connection would result in an invalid CPPN structure
      (e.g., isolating a node, disconnecting graph components vital for
      input-to-output paths), as determined by `validate_cppn`.
    - Removing the connection creates a graph that cannot be topologically sorted
      (though this is less likely if `validate_cppn` passes, as it checks for
      reachability which implies acyclicity is maintained or was already an issue).

    Args:
        cppn: The `FunctionalCPPN` to mutate.
        in_node_id: The ID of the node where the connection to remove starts.
        out_node_id: The ID of the node where the connection to remove ends.

    Returns:
        A new `FunctionalCPPN` instance without the specified connection.
        If the mutation is aborted, the original `cppn` is returned.
    """
    new_connections = [
        conn
        for conn in cppn.connections
        if not (conn.in_node == in_node_id and conn.out_node == out_node_id)
    ]
    if len(new_connections) == len(cppn.connections):
        # print("No such connection found to remove.")
        return cppn # Connection not found

    # Validate the CPPN structure *after* removing the connection.
    if not validate_cppn(cppn.nodes, new_connections):
        # print("Mutation aborted: removing connection would result in an invalid network.")
        return cppn

    # Try to build the new CPPN; this includes a topological sort.
    try:
        return build_cppn(list(cppn.nodes.values()), new_connections)
    except ValueError: # e.g., from topological_sort failure
        # print("Mutation aborted due to topological sort failure after removing connection.")
        return cppn


def mutate_remove_node(cppn: FunctionalCPPN, node_id: int) -> FunctionalCPPN:
    """Removes a node and all its associated connections from the network.

    The mutation is aborted (original `cppn` returned) if:
    - The specified `node_id` does not exist.
    - The node is an `InputNode` or `OutputNode` (these are generally fixed parts
      of the network's interface and not removed by this mutation).
    - Removing the node (and its connections) would result in an invalid CPPN
      structure (e.g., isolating other nodes, breaking paths from all inputs
      to all outputs), as determined by `validate_cppn`.
    - The removal results in a graph that cannot be topologically sorted.

    Args:
        cppn: The `FunctionalCPPN` to mutate.
        node_id: The ID of the node to remove.

    Returns:
        A new `FunctionalCPPN` instance without the specified node and its
        connections. If the mutation is aborted, the original `cppn` is returned.
    """
    if node_id not in cppn.nodes:
        # print("Node does not exist.")
        return cppn

    if isinstance(cppn.nodes[node_id], (InputNode, OutputNode)):
        # print("Cannot remove input or output node.")
        return cppn # Do not remove input/output nodes

    new_nodes_dict = {nid: n for nid, n in cppn.nodes.items() if nid != node_id}
    # If new_nodes_dict becomes empty or lacks inputs/outputs, validate_cppn will catch it.
    if not new_nodes_dict: return cppn


    new_connections_list = [
        conn
        for conn in cppn.connections
        if conn.in_node != node_id and conn.out_node != node_id
    ]

    # Validate the CPPN structure *after* removing the node and its connections.
    # Ensure there are still nodes to validate; otherwise, validate_cppn might behave unexpectedly.
    if not new_nodes_dict or not validate_cppn(new_nodes_dict, new_connections_list):
        # print("Mutation aborted: removing node would result in an invalid network.")
        return cppn

    # Try to build the new CPPN.
    try:
        return build_cppn(list(new_nodes_dict.values()), new_connections_list)
    except ValueError: # e.g., from topological_sort failure
        # print("Mutation aborted due to topological sort failure after removing node.")
        return cppn


def mutate(
    cppn: FunctionalCPPN,
    mutation_probs: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> FunctionalCPPN:
    """Randomly applies one of the available mutation operators to the CPPN.

    The choice of mutation (`mutate_add_node`, `mutate_add_connection`,
    `mutate_remove_connection`, `mutate_remove_node`) is made based on the
    relative probabilities specified in `mutation_probs`.
    If a chosen mutation operation cannot be performed (e.g., trying to remove
    a node from an empty network, or adding a connection that creates a cycle),
    the original CPPN may be returned by the underlying mutation function.

    Args:
        cppn: The `FunctionalCPPN` to mutate.
        mutation_probs: A tuple of four floats representing the relative weights for
            selecting each mutation type:
            (prob_add_node, prob_add_connection, prob_remove_connection, prob_remove_node).
            These weights are normalized internally to form probabilities.

    Returns:
        A new `FunctionalCPPN` instance resulting from the applied mutation.
        If the chosen mutation operator aborts, it returns the original `cppn`.
    """
    # Ensure there's something to mutate, or that mutations can handle empty/small CPPNs.
    # Define mutation types
    mutation_operators = [
        "add_node", "add_connection", "remove_connection", "remove_node"
    ]
    # Normalize probabilities
    total_prob = sum(mutation_probs)
    if total_prob == 0: # Avoid division by zero if all probs are zero
        return cppn # No mutation can be chosen
    normalized_probs = [p / total_prob for p in mutation_probs]

    # Choose a mutation
    chosen_mutation_type = random.choices(mutation_operators, weights=normalized_probs, k=1)[0]

    if chosen_mutation_type == "add_node":
        return mutate_add_node(cppn) # connection_index will be random by default

    elif chosen_mutation_type == "add_connection":
        node_ids = list(cppn.nodes.keys())
        if len(node_ids) < 1: # Need at least one node to pick, ideally 2 for a connection
            return cppn

        # Filter out pairs that are not allowed (e.g., to InputNode, from OutputNode, self-loops)
        possible_starts = [nid for nid in node_ids if not isinstance(cppn.nodes[nid], OutputNode)]
        possible_ends = [nid for nid in node_ids if not isinstance(cppn.nodes[nid], InputNode)]

        if not possible_starts or not possible_ends:
            return cppn

        # Try a few times to find a valid pair to connect
        for _ in range(10): # Attempt to find a valid pair
            in_node_id = random.choice(possible_starts)
            out_node_id = random.choice(possible_ends)
            if in_node_id != out_node_id: # Avoid self-loops here, mutate_add_connection also checks
                # Further checks (like cycle) are in mutate_add_connection itself
                return mutate_add_connection(cppn, in_node_id, out_node_id)
        return cppn # Failed to find a pair after attempts

    elif chosen_mutation_type == "remove_connection":
        if not cppn.connections:
            return cppn
        conn_to_remove = random.choice(cppn.connections)
        return mutate_remove_connection(cppn, conn_to_remove.in_node, conn_to_remove.out_node)

    elif chosen_mutation_type == "remove_node":
        # Only hidden nodes can be removed by mutate_remove_node.
        hidden_node_ids = [
            nid for nid, node in cppn.nodes.items()
            if not isinstance(node, (InputNode, OutputNode))
        ]
        if not hidden_node_ids:
            return cppn
        node_to_remove_id = random.choice(hidden_node_ids)
        return mutate_remove_node(cppn, node_to_remove_id)

    return cppn # Should not be reached if mutation_operators list is correct


def init_cppn(
    input_node_labels: List[str],
    output_node_labels: List[str],
    hidden_activation: str = "sigmoid",
    hidden_aggregation: str = "sum",
) -> FunctionalCPPN:
    """Initializes a basic CPPN structure with one hidden node.

    Creates a CPPN with the specified input nodes, output nodes, and a single
    hidden node (ID 0). Each input node is connected to the hidden node, and the
    hidden node is connected to each output node. All initial connection weights
    are set to 1.0. The hidden node uses the specified activation and aggregation
    functions.

    This function provides a starting point for evolutionary algorithms or other
    generative processes that subsequently modify the CPPN using mutation operators.

    Args:
        input_node_labels: A list of strings, where each string is the label
            for an input node (e.g., `["x", "y", "d"]`).
        output_node_labels: A list of strings, where each string is the label
            for an output node (e.g., `["r", "g", "b"]`).
        hidden_activation: The name of the activation function for the initial
            hidden node (default: "sigmoid").
        hidden_aggregation: The name of the aggregation function for the initial
            hidden node (default: "sum").

    Returns:
        A new `FunctionalCPPN` instance representing this initial network structure.

    Example:
        >>> from jax_cppn.network import init_cppn
        >>> cppn = init_cppn(input_node_labels=["x", "y", "d"], output_node_labels=["r", "g", "b"])
        >>> print(f"Number of nodes: {len(cppn.nodes)}")
        Number of nodes: 7
        >>> print(f"Number of connections: {len(cppn.connections)}")
        Number of connections: 6
        >>> # Node IDs: 0 (hidden), 1,2,3 (inputs), 4,5,6 (outputs) - example assignment
        >>> # Connections: 3 from inputs to hidden, 3 from hidden to outputs
    """
    nodes: List[Node] = []
    connections: List[Connection] = []

    # Central hidden node ID
    hidden_node_id = 0 # By convention for this init function
    hidden_node = Node(
        activation=hidden_activation,
        aggregation=hidden_aggregation,
        node_id=hidden_node_id,
        label=f"H0_{hidden_activation}" # Example label
    )
    nodes.append(hidden_node)

    next_node_id = 1 # Start other node IDs from 1

    # Add input nodes and connect them to the hidden node
    for label in input_node_labels:
        input_id = next_node_id
        nodes.append(InputNode(node_id=input_id, label=label))
        connections.append(Connection(in_node=input_id, out_node=hidden_node_id, weight=1.0))
        next_node_id += 1

    # Add output nodes and connect the hidden node to them
    for label in output_node_labels:
        output_id = next_node_id
        nodes.append(OutputNode(node_id=output_id, label=label, aggregation="sum")) # Default agg for output
        connections.append(Connection(in_node=hidden_node_id, out_node=output_id, weight=1.0))
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
    print(f"Initial CPPN: {cppn_net.nodes.keys()=}, {cppn_net.connections=}")


    # (add_node, add_connection, remove_node, remove_connection)
    # mutation_probs: Tuple[float, float, float, float] = (0.15, 0.6, 0.1, 0.2)
    mutation_probs_tuple: Tuple[float, float, float, float] = (0.15, 0.6, 0.1, 0.2)

    for i in range(50):
        print(f"\nMutation {i+1}")
        cppn_net = mutate(cppn_net, mutation_probs_tuple)
        print(f"Nodes: {sorted(cppn_net.nodes.keys())}")
        # print(f"Connections: {[(c.in_node, c.out_node, round(c.weight,2)) for c in cppn_net.connections]}")
        # Validate after each mutation
        is_valid = validate_cppn(cppn_net.nodes, cppn_net.connections)
        print(f"Is CPPN valid after mutation? {is_valid}")
        if not is_valid:
            print("Warning: Invalid CPPN produced!")
            # Potentially visualize or debug here
            # visualize_cppn_graph(cppn_net, "invalid_cppn")
            break # Stop if network becomes invalid

    print("\nFinal CPPN structure:")
    print(f"Nodes: {cppn_net.nodes}")
    print(f"Connections: {cppn_net.connections}")
    print(f"Topological order: {cppn_net.topo_order}")


    output = forward_cppn(cppn_net, inputs)
    plot_output(x_coords, y_coords, output["out"])
    visualize_cppn_graph(cppn_net, "final_cppn_graph")

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
