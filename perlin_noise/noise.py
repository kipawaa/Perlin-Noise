# generates a perlin noise map
from functools import lru_cache
from random import randint

import numpy as np
from PIL import Image


def fade(x):
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a, b, weight):
    return a + weight * (b - a)


# sub_w is the number of sub-boxes used in generation across the width of the map, sub_h the same for the height of the map
def noise(width: int, height: int, sub_w: int, sub_h: int) -> np.ndarray:
    """returns a perlin noise height map with values from 0 to 255"""
    # number of pixels in the height and width of one sub-box
    grid_width = width // sub_w
    grid_height = height // sub_h

    # creates random angles that are then used to generate the
    # random gradient vectors from the corners of each sub-box
    angles = 2 * np.pi * np.random.rand(sub_h + 1, sub_w + 1)
    gradients = np.empty((sub_h + 1, sub_w + 1, 2), dtype=angles.dtype)
    np.cos(angles, out=gradients[..., 0])
    np.sin(angles, out=gradients[..., 1])

    # pre-compute offsets and grid coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid_indices, y_grid_indices = np.meshgrid(x // grid_width, y // grid_height)
    x_offsets, y_offsets = np.meshgrid(x / grid_width % 1, y / grid_height % 1)
    x_weights = fade(x_offsets)
    y_weights = fade(y_offsets)

    gradient_vectors = gradients[
        y_grid_indices[..., np.newaxis] + np.array([0, 0, 1, 1]),
        x_grid_indices[..., np.newaxis] + np.array([0, 1, 0, 1]),
        :,
    ]
    distance_vectors = np.stack(
        [
            x_offsets[..., np.newaxis] + np.array([0, -1, 0, -1]),
            y_offsets[..., np.newaxis] + np.array([0, 0, -1, -1]),
        ],
        axis=-1,
    )

    dots = np.sum(gradient_vectors * distance_vectors, axis=-1)

    top_lerped_vector = lerp(dots[:, :, 0], dots[:, :, 1], x_weights)
    bottom_lerped_vector = lerp(dots[:, :, 2], dots[:, :, 3], x_weights)
    noise_array = lerp(top_lerped_vector, bottom_lerped_vector, y_weights)

    return noise_array


def normalize(noise_array):
    """Normalize a noise array to [0, 255]."""
    noise_array -= noise_array.min()
    return (noise_array / noise_array.max() * 255).astype(np.uint8)


def get_octave_noise(
    width: int, height: int, n_octaves: int, grid_width: int, grid_height: int, factor: int = 2
):
    """Return octave stacked noise."""
    # Pre-allocate the output 2D buffer once
    total_noise = np.zeros((height, width), dtype=np.float32)

    for i in range(n_octaves):
        sub_w = grid_width * (factor**i)
        sub_h = grid_height * (factor**i)

        if sub_w >= width or sub_h >= height:
            break

        octave_data = noise(width, height, sub_w, sub_h)
        total_noise += octave_data / (factor**i)

    return normalize(total_noise)


if __name__ == "__main__":
    noise = get_octave_noise(2000, 1000, 4, 5, 2)
    Image.fromarray(noise).show()
