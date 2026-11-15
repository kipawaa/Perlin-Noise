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

    # TODO: add layer blending


def get_snowy_pixel(height):
    # probabilistically use the noise colour (bright) but scaled within range(128, 255)
    if random() > 1 - (height / 255):
        return np.asarray(
            (
                (height - 200) * 4.5,
                (height - 200) * 4.5,
                (height - 200) * 4.5,
            )
        )

    # default stone value
    return np.asarray((112, 112, 112))


def get_stone_pixel(height):
    return np.asarray((112, 112, 112))


def makeImage(height_map, config: LayerConfig = LayerConfig()):
    # stores the colour values of the map
    map_height, map_width = height_map.shape
    img = np.zeros((map_height, map_width, 3), "uint8")

    for x in range(map_width):
        for y in range(map_height):
            # probabilistic snowy caps and rocky mountains
            height = height_map[y, x]
            if 200 <= height <= config.snow:
                img[y, x] = get_snowy_pixel(height)

            # stony mountains
            if height <= config.stone:
                img[y, x] = get_stone_pixel(height)

            # attempt at making things pretty green colours
            if height <= config.forest:
                img[y, x] = lerp(np.asarray((27, 48, 41)), np.asarray((70, 90, 72)), (height / 255))

            # watery depths
            if height <= config.water:
                img[y, x] = lerp(np.asarray((15, 30, 58)), np.asarray((8, 24, 49)), (height / 255))

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
