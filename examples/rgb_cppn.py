# %%
from jax_cppn.network import init_cppn, mutate, forward_cppn
from jax_cppn.vis import visualize_cppn_graph
import jax.numpy as jnp
import matplotlib.pyplot as plt

cppn_net = init_cppn(["x", "y", "d"], ["r", "g", "b"])

# randomly mutate the CPPN
for _ in range(30):
    cppn_net = mutate(cppn_net)

# %%
res = 128
x_coords = jnp.linspace(-1, 1, res)
y_coords = jnp.linspace(-1, 1, res)
XX, YY = jnp.meshgrid(x_coords, y_coords)
# radial distance to the centre of the image
DD = jnp.sqrt(XX**2 + YY**2)
inputs = {"x": XX, "y": YY, "d": DD}

output = forward_cppn(cppn_net, inputs)
img_output = jnp.stack([output["r"], output["g"], output["b"]], axis=2)
plt.imshow(img_output)
plt.show()
visualize_cppn_graph(cppn_net)

# %%
