# generates a map-style image using perlin noise

from dataclasses import dataclass
from random import randint, random

import numpy as np
from PIL import Image

from perlin_noise.noise import lerp, noise


@dataclass(frozen=True)
class LayerConfig:
    water: int = 60
    forest: int = 190
    stone: int = 245
    snow: int = 255


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


# width and height define the resolution of the array, minNoise and maxNoise define how many boxes (the corners of which are used to generate vectors that create the noise) are used to create the noise
def generateHeights(width, height, numLayers, minNoiseRes=2, maxNoiseRes=15):
    heightMap = np.zeros((height, width))
    for i in range(numLayers):
        # used to determine the noise resolution
        numBoxes = (maxNoiseRes // numLayers) + minNoiseRes
        # generates a new layer of noise and adds the value to the height map (scaled)
        heightMap += (1 - (i / numLayers)) * noise(
            width, height, numBoxes * width // height, numBoxes * height // width
        )

    # normalizes the map to the range(0, 255)
    heightMap -= heightMap.min()
    heightMap = (heightMap / heightMap.max() * 255).astype(np.uint8)
    return heightMap


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
    heightMap = generateHeights(width, height, 3)
    img = makeImage(heightMap)

    # creates an image from the map
    print("generating image")
    Image.fromarray(img).show()

    print("complete")
