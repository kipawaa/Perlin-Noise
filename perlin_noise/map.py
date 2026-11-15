# generates a map-style image using perlin noise

from dataclasses import dataclass
from random import randint, random

import numpy as np
from PIL import Image

from perlin_noise.layer import ForestLayer, Layer, SnowLayer, StoneLayer, WaterLayer
from perlin_noise.noise import get_octave_noise, lerp


def makeImage(height_map, layers: list[Layer]):
    # stores the colour values of the map
    image = np.zeros((*height_map.shape, 3), "uint8")

    for layer in layers:
        image = layer.apply(height_map, image)

    return image


if __name__ == "__main__":
    # resolution of the map
    width = 1920
    height = 1080

    # generates the heights
    print("generating heights")
    heightMap = get_octave_noise(width, height, 3, 10, 5)
    img = makeImage(heightMap, layers=[SnowLayer(), StoneLayer(), ForestLayer(), WaterLayer()])

    # creates an image from the map
    print("generating image")
    Image.fromarray(img).show()

    print("complete")
