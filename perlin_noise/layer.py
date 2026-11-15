import numpy as np

from perlin_noise.noise import lerp


class Layer:
    def apply(self, height_map, image):
        return NotImplemented


class SnowLayer(Layer):
    def apply(self, height_map, image):
        # probabilistic snowy caps and rocky mountains
        map_height, map_width = height_map.shape
        snow_layer = 245 <= height_map
        rands = np.random.rand(map_height, map_width)
        snow_mask = snow_layer & (rands > (1 - height_map / 255))
        image[snow_mask] = np.stack(
            [height_map[snow_mask], height_map[snow_mask], height_map[snow_mask]], axis=-1
        )
        return image


class StoneLayer(Layer):
    def apply(self, height_map, image):
        stone_mask = height_map <= 245
        image[stone_mask] = [112, 112, 112]
        return image


class ForestLayer(Layer):
    def apply(self, height_map, image):
        forest_mask = height_map <= 190
        image[forest_mask] = lerp(
            np.asarray((27, 48, 41)),
            np.asarray((70, 90, 72)),
            (height_map / 255)[..., np.newaxis],
        )[forest_mask]
        return image


class WaterLayer(Layer):
    def apply(self, height_map, image):
        # watery depths
        water_mask = height_map <= 60
        image[water_mask] = lerp(
            np.asarray((8, 24, 49)),
            np.asarray((15, 30, 58)),
            (height_map / 255)[..., np.newaxis],
        )[water_mask]

        return image
