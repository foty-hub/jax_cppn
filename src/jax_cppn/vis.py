import networkx as nx
import matplotlib.pyplot as plt


def visualize_cppn_network(cppn_net):
    G = nx.DiGraph()

    # Build a set of nodes that have outgoing connections.
    # (i.e. every node that appears as a source (in_node) in any connection)
    source_nodes = {conn.in_node for conn in cppn_net.connections}

    # Create labels for each node:
    node_labels = {}
    for node_id, node in cppn_net.nodes.items():
        if len(cppn_net.incoming[node_id]) == 0:
            # No incoming connections => input node.
            node_labels[node_id] = node.label or "in"
        elif node_id not in source_nodes:
            # Not a source in any connection => output node.
            node_labels[node_id] = node.label or "out"
        else:
            node_labels[node_id] = node.label or node.act_str

        # Add node with label (we'll override label in drawing with our dict)
        G.add_node(node_id, label=node_labels[node_id])

    # Add edges with weight as an attribute (to control line thickness).
    for conn in cppn_net.connections:
        G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)

    # Use the layered_layout defined below.
    pos = layered_layout(cppn_net)

    # Draw the nodes.
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1000)

    # Draw node labels using our custom labels.
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Calculate edge widths from the absolute value of the weight.
    widths = [abs(data["weight"]) for u, v, data in G.edges(data=True)]

    # Draw edges with arrowheads.
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle="->", arrowsize=20, width=widths, node_size=1000
    )

    plt.title("CPPN Network Graph (Layered Layout)")
    plt.axis("off")
    plt.show()


def layered_layout(cppn_net):
    """
    Assigns each node a (x, y) coordinate based on a simple topological-layer layout.
    - Input nodes (no incoming edges) start at layer 0 (the bottom).
    - Each child node is placed one layer higher than its deepest parent.
    - The highest layer is thus at the top (largest y-value).
    """

    # 1) Determine each node's layer based on the topological order.
    layer_of = {}

    # Initialize input nodes (no incoming edges) to layer 0.
    for node_id in cppn_net.topo_order:
        if not cppn_net.incoming[node_id]:  # no incoming edges
            layer_of[node_id] = 0

    # Assign layers in topological order:
    for node_id in cppn_net.topo_order:
        current_layer = layer_of.get(node_id, 0)
        # Find all children of this node.
        children = [
            conn.out_node for conn in cppn_net.connections if conn.in_node == node_id
        ]
        for child in children:
            # The child's layer is at least (current_layer + 1)
            layer_of[child] = max(layer_of.get(child, 0), current_layer + 1)

    # 2) Group nodes by their layer.
    layers = {}
    for node_id, layer in layer_of.items():
        layers.setdefault(layer, []).append(node_id)

    # 3) Build the (x, y) positions.
    #    We'll leave layer 0 at the bottom, and the highest layer at the top.
    pos = {}
    for layer, nodes_in_layer in layers.items():
        y = layer  # direct use of layer: input at bottom (layer=0), output at top
        count = len(nodes_in_layer)
        for i, node_id in enumerate(nodes_in_layer):
            x = i - (count - 1) / 2.0  # e.g., if 3 nodes, x = -1, 0, +1
            pos[node_id] = (x, y)

    return pos
