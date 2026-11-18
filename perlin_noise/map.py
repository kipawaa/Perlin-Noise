# generates a map-style image using perlin noise

import tkinter as tk
from dataclasses import dataclass
from random import randint, random

import numpy as np
from PIL import Image

from perlin_noise.layer import (
    FlatDarkForestLayer,
    FlatLightForestLayer,
    FlatMediumForestLayer,
    ForestLayer,
    Layer,
    MediumWaterLayer,
    SecondFlatMediumForestLayer,
    ShallowWaterLayer,
    SnowLayer,
    StoneLayer,
    WaterLayer,
)
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
    heightMap = get_octave_noise(width, height, 4, 10, 5)
    img = makeImage(
        heightMap,
        layers=[
            SnowLayer(),
            StoneLayer(),
            FlatDarkForestLayer(),
            FlatMediumForestLayer(),
            SecondFlatMediumForestLayer(),
            FlatLightForestLayer(),
            ShallowWaterLayer(),
            MediumWaterLayer(),
            WaterLayer(),
        ],
    )

    # creates an image from the map
    Image.fromarray(img).show()
