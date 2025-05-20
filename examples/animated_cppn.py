import argparse

import imageio
import jax.numpy as jnp
import numpy as np

from jax_cppn.network import init_cppn, mutate, forward_cppn


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate an animated CPPN GIF.")
    parser.add_argument(
        "-o",
        "--output",
        default="animated_cppn.gif",
        help="Output filename for the GIF.",
    )
    return parser.parse_args()


def initialize_network():
    """Initializes the CPPN."""
    input_names = ["x", "y", "d", "t"]
    output_names = ["r", "g", "b"]
    return init_cppn(input_names, output_names)


def mutate_network(cppn_net, num_mutations=30):
    """Mutates the CPPN multiple times."""
    for _ in range(num_mutations):
        cppn_net = mutate(cppn_net)
    return cppn_net


def generate_frames(cppn_net, img_res=128, num_frames=50):
    """Generates animation frames from the CPPN."""
    x_coords = jnp.linspace(-1, 1, img_res)
    y_coords = jnp.linspace(-1, 1, img_res)
    XX, YY = jnp.meshgrid(x_coords, y_coords)
    DD = jnp.sqrt(XX**2 + YY**2)

    generated_frames_list = []
    for frame_idx in range(num_frames):
        time_value = jnp.sin(2 * jnp.pi * frame_idx / num_frames)
        TT = jnp.full_like(XX, time_value)

        inputs = {"x": XX, "y": YY, "d": DD, "t": TT}
        output_values = forward_cppn(cppn_net, inputs)

        img = jnp.stack(
            [output_values["r"], output_values["g"], output_values["b"]], axis=2
        )
        img = jnp.clip(img, 0.0, 1.0)
        generated_frames_list.append(img)

    return generated_frames_list


def save_animation_gif(frames_list, filename, fps=20):
    """Saves the list of JAX image arrays as a GIF."""
    processed_frames_for_gif = []
    for jax_frame in frames_list:
        np_frame = np.array(jax_frame)
        uint8_frame = (np_frame * 255).astype(np.uint8)
        processed_frames_for_gif.append(uint8_frame)
    imageio.mimsave(filename, processed_frames_for_gif, fps=fps)


def main(args):
    """Runs the main script logic."""
    print(f"Output filename: {args.output}")

    cppn = initialize_network()
    print(f"CPPN initialized")

    cppn = mutate_network(cppn)  # Uses default 30 mutations
    print(f"CPPN mutated 30 times.")

    frames_list = generate_frames(cppn)  # Uses default 128 resolution, 50 frames
    print(f"Generated {len(frames_list)} animation frames.")

    save_animation_gif(frames_list, args.output, fps=20)  # Uses default 20 FPS
    print(f"Animation saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
