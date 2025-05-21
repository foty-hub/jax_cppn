"""Visualization utilities for Compositional Pattern Producing Networks (CPPNs).

This module provides functions to visualize the structure of a CPPN graph and
to plot its output, typically as a 2D image or pattern. It uses `networkx`
for graph manipulation and `matplotlib` for plotting.
"""
import networkx as nx
import matplotlib.pyplot as plt
from jax_cppn.node import InputNode, OutputNode
from jax_cppn.network import FunctionalCPPN # For type hinting and examples
import jax.numpy as jnp # For examples


# TODO: sometimes the graph vis fails to show inputs on the first layer


def visualize_cppn_graph(cppn_net: FunctionalCPPN) -> None:
    """Visualizes the structure of a CPPN using `networkx` and `matplotlib`.

    This function generates a visual representation of the CPPN graph,
    displaying nodes, connections, and their attributes.
    - Nodes are typically labeled with their activation function or a custom label.
    - Connections (edges) are drawn with widths proportional to their weights.
    - The layout attempts to position nodes in layers based on their topological order.

    Args:
        cppn_net: The `FunctionalCPPN` instance to visualize.

    Example:
        For a usage example, refer to `examples/rgb_cppn.py`.
        A conceptual example:
        >>> from jax_cppn.network import init_cppn
        >>> from jax_cppn.vis import visualize_cppn_graph
        >>> # cppn = init_cppn(["x"], ["out"]) # Create a simple CPPN
        >>> # visualize_cppn_graph(cppn) # This would display a plot window
        >>> print("Visualization function would be called here.") # Placeholder
        Visualization function would be called here.
    """
    G = nx.DiGraph()

    # Create labels for each node:
    node_labels = {}
    for node_id, node in cppn_net.nodes.items():
        # Use custom label if present, otherwise default to activation string
        label = node.label if node.label else node.act_str
        node_labels[node_id] = label
        G.add_node(node_id, label=label) # Add node with its resolved label

    # Add edges with weight as an attribute.
    for conn in cppn_net.connections:
        G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)

    # Use the layered_layout defined in this module.
    pos = layered_layout(cppn_net)

    # Draw the nodes.
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000)

    # Draw node labels using the prepared labels.
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Calculate edge widths from the absolute value of the weight.
    # Ensure there are edges before trying to access data['weight']
    widths = [abs(G.edges[u, v]['weight']) for u, v in G.edges()] if G.edges() else []


    # Draw edges with arrowheads.
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="->", arrowsize=20, width=widths, node_size=1000
    )

    plt.title("CPPN Network Graph (Layered Layout)")
    plt.axis("off") # Turn off the axis numbers and ticks
    plt.show() # Display the plot


def layered_layout(cppn_net: FunctionalCPPN) -> dict:
    """Computes node positions for a layered graph layout of a CPPN.

    This layout algorithm assigns (x, y) coordinates to each node based on its
    topological layer in the CPPN.
    1. Layer Assignment:
        - Input nodes (nodes with no incoming connections or explicitly typed as `InputNode`)
          are initially assigned to layer 0.
        - Nodes are processed in topological order. The layer of a node is determined
          as `max(parent_layer) + 1` for all its parents.
        - Output nodes are explicitly moved to the highest layer number determined
          among all nodes to ensure they appear at the top/end of the visualization.
    2. Positioning:
        - Nodes within the same layer are spread out horizontally (x-coordinate).
        - The y-coordinate corresponds to the layer number, placing layer 0 at the
          bottom and higher layers progressively upwards.

    Args:
        cppn_net: The `FunctionalCPPN` instance for which to compute the layout.

    Returns:
        A dictionary where keys are node IDs and values are (x, y) tuples
        representing the position of each node.
    """
    layer_of: dict[int, int] = {}

    # Initialize input nodes to layer 0.
    for node_id in cppn_net.topo_order:
        # A more robust check for input nodes: those typed as InputNode
        # or nodes with no incoming connections if not explicitly typed (though FunctionalCPPN structure implies types).
        if isinstance(cppn_net.nodes[node_id], InputNode):
            layer_of[node_id] = 0
        # Fallback for nodes that might act as inputs but are not InputNode instances
        # and are at the start of topo_order with no incoming connections from cppn_net.connections
        elif not cppn_net.incoming.get(node_id): # No incoming connections
             layer_of.setdefault(node_id, 0)


    # Assign layers in topological order:
    for node_id in cppn_net.topo_order:
        current_layer = layer_of.get(node_id, 0) # Default to 0 if somehow not set (e.g. disconnected node)
        # Find all children of this node using the connections list.
        children = [
            conn.out_node for conn in cppn_net.connections if conn.in_node == node_id
        ]
        for child_id in children:
            layer_of[child_id] = max(layer_of.get(child_id, 0), current_layer + 1)

    # Ensure output nodes are on the final (highest) layer.
    if layer_of: # Proceed only if layers were assigned
        max_layer = max(layer_of.values()) if layer_of else 0
        for node_id in cppn_net.nodes: # Iterate all nodes to find OutputNodes
            if isinstance(cppn_net.nodes[node_id], OutputNode):
                layer_of[node_id] = max_layer
    else: # Handle empty or minimally structured cppn_net
        if not cppn_net.nodes: return {} # No nodes, no positions
        # If nodes exist but no layers assigned (e.g. single node), assign all to layer 0
        for node_id in cppn_net.nodes:
            layer_of.setdefault(node_id,0)


    # Group nodes by their layer.
    layers: dict[int, list[int]] = {}
    for node_id, layer_val in layer_of.items():
        layers.setdefault(layer_val, []).append(node_id)

    # Build the (x, y) positions.
    pos: dict[int, tuple[float, float]] = {}
    # Calculate y-coordinates such that layer 0 is at the bottom (y=0)
    # and higher layers increase y. Matplotlib default is y increasing upwards.
    # Max layer value for normalization if needed, but direct use is fine.
    # y_max_val = max(layers.keys()) if layers else 0

    for layer_val, nodes_in_layer in layers.items():
        y = float(layer_val) # Layer number directly as y-coordinate
        count = len(nodes_in_layer)
        # Spread nodes horizontally: x = -width/2 to +width/2
        for i, node_id in enumerate(nodes_in_layer):
            # Calculate x to center the nodes in the layer around x=0
            x = (i - (count - 1) / 2.0) * 1.0 # Multiply by spacing factor if needed
            pos[node_id] = (x, y)
    return pos


def plot_output(x_coords: jnp.ndarray, y_coords: jnp.ndarray, output: jnp.ndarray) -> None:
    """Generates a 2D contour plot of a CPPN's output.

    This function is typically used when the CPPN generates a 2D pattern (e.g., an image).
    It uses `matplotlib.pyplot.contourf` to create a filled contour plot.
    The `x_coords` and `y_coords` define the grid onto which the `output` values are mapped.

    Args:
        x_coords: A 1D JAX numpy array of x-coordinates (e.g., from `jnp.linspace`).
        y_coords: A 1D JAX numpy array of y-coordinates.
        output: A 2D JAX numpy array representing the CPPN's output values over the
            grid defined by `x_coords` and `y_coords` (typically after meshgrid).
            The shape of `output` should correspond to `(len(y_coords), len(x_coords))`.

    Example:
        For a usage example, see the `if __name__ == "__main__":` block in
        `src/jax_cppn/network.py`.
        A conceptual example:
        >>> import jax.numpy as jnp
        >>> from jax_cppn.vis import plot_output
        >>> x = jnp.linspace(-1, 1, 3) # Small example for doctest
        >>> y = jnp.linspace(-1, 1, 3)
        >>> XX, YY = jnp.meshgrid(x, y)
        >>> Z = jnp.sin(XX*jnp.pi) * jnp.cos(YY*jnp.pi) # Example 2D output
        >>> # plot_output(x, y, Z) # This would display a plot window
        >>> print("Plotting function would be called here.") # Placeholder
        Plotting function would be called here.
    """
    plt.figure()
    # contourf expects X, Y, Z where Z is (Y.shape[0], X.shape[0])
    # x_coords, y_coords are 1D. output is 2D, e.g. (len(y_coords), len(x_coords))
    plt.contourf(x_coords, y_coords, output, cmap="gray", levels=256)
    plt.xlabel("x")
    plt.ylabel("y") # Changed from "output" to "y" for clarity, as output is Z
    plt.title("CPPN Network Output")
    plt.colorbar(label="Output Value") # Add a colorbar for context
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.show()
