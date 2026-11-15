# generates a map-style image using perlin noise

from dataclasses import dataclass
from random import randint, random

import numpy as np
from PIL import Image

from perlin_noise.noise import get_octave_noise, lerp


@dataclass(frozen=True)
class LayerConfig:
    # layers
    water: int = 60
    forest: int = 190
    stone: int = 245
    snow: int = 255

    # layer blending radius
    radius: int = 5


def makeImage(height_map, config: LayerConfig = LayerConfig()):
    # stores the colour values of the map
    map_height, map_width = height_map.shape
    img = np.zeros((map_height, map_width, 3), "uint8")

    # probabilistic snowy caps and rocky mountains
    snow_layer = (200 <= height_map) & (height_map <= config.snow)
    rands = np.random.rand(map_height, map_width)
    snow_mask = snow_layer & (rands > (1 - height_map / 255))
    img[snow_mask] = np.stack(
        [height_map[snow_mask], height_map[snow_mask], height_map[snow_mask]], axis=-1
    )

    # stony mountains
    stone_mask = height_map <= config.stone
    img[stone_mask] = [112, 112, 112]

    # forest layer
    forest_mask = height_map <= config.forest
    img[forest_mask] = lerp(
        np.asarray((27, 48, 41)), np.asarray((70, 90, 72)), (height_map / 255)[..., np.newaxis]
    )[forest_mask]

    # watery depths
    water_mask = height_map <= config.water
    img[water_mask] = lerp(
        np.asarray((15, 30, 58)), np.asarray((8, 24, 49)), (height_map / 255)[..., np.newaxis]
    )[water_mask]

    return img


if __name__ == "__main__":
    # resolution of the map
    width = 1920
    height = 1080

    # generates the heights
    print("generating heights")
    heightMap = get_octave_noise(width, height, 3, 10, 5)
    img = makeImage(heightMap)

    # creates an image from the map
    print("generating image")
    Image.fromarray(img).show()

    print("complete")
