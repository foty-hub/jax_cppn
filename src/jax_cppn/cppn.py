# %%
import jax
import jax.numpy as jnp
from typing import Literal, Callable
import matplotlib.pyplot as plt
from jax.scipy.stats import multivariate_normal
from collections import deque, defaultdict

# %%
# rough structure of a CPPN, from https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/cppnNEAT.pdf
# start with 2D inputs
# %%
_ACTIVATION_FUNCTIONS: list[str] = [
    "gaussian",
    "sigmoid",
    "sin",
    "linear",
]

# Define the mean and covariance for a 2D Gaussian.


# Plot the PDF as a contour plot.
def plot_cppn_out(XX: jnp.ndarray, YY: jnp.ndarray, values: jnp.ndarray) -> None:
    plt.figure()
    plt.contourf(XX, YY, values, levels=50, cmap="gray")
    plt.title("2D Gaussian PDF")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="PDF value")
    plt.show()


def main(): ...


if __name__ == "__main__":
    main()


# %%
class GaussianNode:
    def __init__(
        self,
        mean: jnp.ndarray,
        cov_vec: jnp.ndarray,
    ) -> None:
        """Intialise a Gaussian CPPN node.

        Parameters:
        """
        self.mean = mean
        self.n_inputs = len(mean)
        self.cov_vec = cov_vec
        # cov len parameterises the lower triangular matrix - should be
        # length n(n+1)/2, where n is the number of inputs
        expected_cov_len = int(self.n_inputs * (self.n_inputs + 1) / 2)
        assert len(cov_vec) == expected_cov_len, (
            f"Expected the covariance vector to have {expected_cov_len} elements, rather than {len(cov_vec)}"
        )

        cov = self.vec_to_cov(cov_vec)
        self.vpdf = jax.vmap(lambda x: multivariate_normal.pdf(x, mean, cov))

    def __call__(
        self,
        points: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Evaluate the PDF of an n-dimensional Gaussian at each given point.

        Parameters:
        points: jnp.ndarray of shape (N, n) where each row is an n-dimensional point.

        Returns:
        jnp.ndarray of shape (N,) with the PDF evaluated at each point.
        """
        # Vectorize the PDF computation over the first axis of points.
        return self.vpdf(points)

    def vec_to_cov(self, vec: jnp.ndarray) -> jnp.ndarray:
        # Convert compact vector representation to a Cholesky-decomposed
        # covariance matrix
        L = jnp.zeros((self.n_inputs, self.n_inputs))
        indices = jnp.tril_indices(self.n_inputs)
        L = L.at[indices].set(vec)
        return L @ L.T


# %%

# Create a 2D grid of points.
x_coords = jnp.linspace(-5, 5, 100)
y_coords = jnp.linspace(-5, 5, 100)
XX, YY = jnp.meshgrid(x_coords, y_coords)
points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)


mean = jnp.array([0.0, 0.0])
vec = jnp.array([1.0, 0.2, 1.0])
node = GaussianNode(mean, vec)
pdf_vals = node(points)
pdf_grid = pdf_vals.reshape(XX.shape)


# %%
plot_cppn_out(XX, YY, pdf_grid)
# %%
points.shape, pdf_vals.shape
# %%


# Define a simple connection class.
class Connection:
    def __init__(self, source: str, target: str, weight: float) -> None:
        self.source = source  # ID of the source node.
        self.target = target  # ID of the target node.
        self.weight = weight  # Weight on the connection.


# Network class that composes multiple nodes.
class Network:
    def __init__(
        self,
        nodes: dict[str, Callable[[jnp.ndarray], jnp.ndarray]],
        connections: list[Connection],
        input_nodes: list[str],
        output_nodes: list[str],
    ) -> None:
        """
        Parameters:
            nodes: A dictionary mapping a unique node identifier (e.g., a string)
                   to a callable node function (like your GaussianNode instance or others).
            connections: A list of Connection objects describing how nodes are linked.
            input_nodes: List of node IDs designated as inputs.
            output_nodes: List of node IDs designated as outputs.
        """
        self.nodes = nodes
        self.connections = connections
        self.input_nodes = set(input_nodes)
        self.output_nodes = output_nodes

        # Build a mapping from node ID to a list of incoming connections.
        self.incoming = defaultdict(list)
        for conn in self.connections:
            self.incoming[conn.target].append((conn.source, conn.weight))

        # Compute a topological order for the nodes. This is required for proper forward propagation.
        self.topological_order = self._compute_topological_order()

    def _compute_topological_order(self) -> list[str]:
        """Kahn's algorithm to compute a topological order of nodes in the DAG."""
        # Initialize in-degrees for every node.
        in_degree = {node: 0 for node in self.nodes.keys()}
        for conn in self.connections:
            in_degree[conn.target] += 1

        # Start with nodes having zero in-degree (input nodes typically).
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            # For every connection from current, decrease the in-degree of the target.
            for conn in self.connections:
                if conn.source == current:
                    in_degree[conn.target] -= 1
                    if in_degree[conn.target] == 0:
                        queue.append(conn.target)

        if len(order) != len(self.nodes):
            raise ValueError("The network graph contains a cycle!")
        return order

    def __call__(self, inputs: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
        # Dictionary to hold computed outputs for each node.
        node_outputs = {}

        # Assign external inputs.
        for node in self.input_nodes:
            if node not in inputs:
                raise ValueError(f"Input for node '{node}' is missing.")
            node_outputs[node] = inputs[node]

        # Propagate through nodes in topological order.
        for node in self.topological_order:
            if node in self.input_nodes:
                continue

            # For nodes that need feature concatenation (like 'combine'),
            # gather outputs in a list instead of summing them.
            incoming_values = []
            for source, weight in self.incoming.get(node, []):
                # Here, each incoming value is weighted.
                incoming_values.append(weight * node_outputs[source])
            # Instead of addition, concatenate along the last axis.
            aggregated_input = jnp.concatenate(incoming_values, axis=-1)
            node_outputs[node] = self.nodes[node](aggregated_input)

        return {node: node_outputs[node] for node in self.output_nodes}


# Example usage:
# Assume you have already defined some node functions such as GaussianNode, SinNode, etc.
# For illustration, let’s say we have:
#   nodes = {
#       "in": lambda x: x,  # identity function for input node
#       "gauss": GaussianNode(...),
#       "sin": lambda x: jnp.sin(x)
#   }
# And connections:
#   connections = [
#       Connection("in", "gauss", 1.0),
#       Connection("gauss", "sin", 0.5)
#   ]
# Input nodes and output nodes would be defined accordingly.
# === Example instantiation and network call ===
# %%
# Instantiate the Gaussian node with a 2D mean and covariance vector.
gauss_node = GaussianNode(
    mean=jnp.array([0.0, 0.0]), cov_vec=jnp.array([5.0, 0.2, 5.0])
)

gauss_node2 = GaussianNode(
    mean=jnp.array([1.0, 2.0]), cov_vec=jnp.array([3.0, 0.5, 1.0])
)


# Define nodes: an input node, the Gaussian node, and a sine node.
nodes = {
    "x": lambda x: x,  # returns x-coordinates
    "y": lambda y: y,  # returns y-coordinates
    # Later, a node to combine them
    "combine": lambda inp: jnp.stack(inp, axis=-1),
    "gauss": GaussianNode(
        mean=jnp.array([0.0, 0.0]), cov_vec=jnp.array([1.0, 0.2, 1.0])
    ),
    "sin": lambda x: jnp.sin(x),
    "output": lambda x: x,
}

# Define connections:
#   "input" feeds into "gauss" with weight 1.0.
#   "gauss" feeds into "sin" with weight 0.5.
connections = [
    Connection(source="x", target="combine", weight=1.0),
    Connection(source="y", target="combine", weight=1.0),
    Connection(source="combine", target="gauss", weight=1.0),
    Connection(source="gauss", target="sin", weight=1.0),
    Connection(source="sin", target="output", weight=1.0),
]

# Specify input and output nodes.
input_nodes = ["x", "y"]
output_nodes = ["output"]

# Instantiate the network.
net = Network(nodes, connections, input_nodes, output_nodes)

# Create sample input: a set of 100 2D points.
# Create a 2D grid of points.
x_coords = jnp.linspace(-5, 5, 100)
y_coords = jnp.linspace(-5, 5, 100)
XX, YY = jnp.meshgrid(x_coords, y_coords)
points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

# points = jnp.array(
#     [[x, y] for x in jnp.linspace(-5, 5, 100) for y in jnp.linspace(-5, 5, 100)]
# )

# Propagate the input through the network.
outputs = net({"x": XX.ravel(), "y": YY.ravel()})

# Print the output from the "sin" node.
print("Output of the network (from 'sin' node):")
print(outputs["output"])
# %%
grid_vals = outputs["output"].reshape(100, 100)
plot_cppn_out(XX, YY, grid_vals)
# %%
net.nodes

# %%
