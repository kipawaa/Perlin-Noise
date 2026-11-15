import numpy as np

from perlin_noise.noise import lerp

Colour = tuple[int, int, int]


class Layer:
    def apply(
        self,
        height_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        image: np.ndarray[Colour, np.dtype[np.uint8]],
    ):
        return NotImplemented


class SnowLayer(Layer):
    def __init__(
        self,
        min_height: int = 245,
        max_height: int = 255,
    ):
        self.min_height: int = min_height
        self.max_height: int = max_height

    def apply(
        self,
        height_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        image: np.ndarray[Colour, np.dtype[np.uint8]],
    ):
        # probabilistic snowy caps and rocky mountains
        map_height, map_width = height_map.shape
        snow_layer = self.min_height <= height_map
        rands = np.random.rand(map_height, map_width)
        snow_mask = snow_layer & (rands > (1 - height_map / self.max_height))
        image[snow_mask] = np.stack(
            [height_map[snow_mask], height_map[snow_mask], height_map[snow_mask]], axis=-1
        )
        return image


class StoneLayer(Layer):
    def __init__(
        self,
        stone_colour: Colour | None = None,
        max_height: int = 245,
        blend_down: bool = True,
        blend_radius: int = 5,
    ):
        self.stone_colour = np.asarray(stone_colour or (112, 112, 112))
        self.max_height = max_height
        self.blend_down = blend_down
        self.blend_radius = blend_radius

    def apply(
        self,
        height_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        image: np.ndarray[Colour, np.dtype[np.uint8]],
    ):
        stone_mask = height_map <= self.max_height
        image[stone_mask] = self.stone_colour
        return image


class ForestLayer(Layer):
    def __init__(
        self,
        dark_forest: Colour | None = None,
        light_forest: Colour | None = None,
        max_height: int = 190,
    ):
        self.dark_forest = np.asarray(dark_forest or (27, 48, 41))
        self.light_forest = np.asarray(light_forest or (70, 90, 72))
        self.max_height = max_height

    def apply(
        self,
        height_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        image: np.ndarray[Colour, np.dtype[np.uint8]],
    ):
        forest_mask = height_map <= self.max_height
        image[forest_mask] = lerp(
            self.dark_forest,
            self.light_forest,
            (height_map / self.max_height)[..., np.newaxis],
        )[forest_mask]
        return image


class WaterLayer(Layer):
    def __init__(
        self,
        dark_water: Colour | None = None,
        light_water: Colour | None = None,
        max_height: int = 60,
    ):
        self.dark_water = np.asarray(dark_water or (8, 24, 49))
        self.light_water = np.asarray(light_water or (15, 30, 58))
        self.max_height = max_height

    def apply(
        self,
        height_map: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
        image: np.ndarray[Colour, np.dtype[np.uint8]],
    ):
        # watery depths
        water_mask = height_map <= self.max_height
        image[water_mask] = lerp(
            self.dark_water,
            self.light_water,
            (height_map / self.max_height)[..., np.newaxis],
        )[water_mask]

        return image
