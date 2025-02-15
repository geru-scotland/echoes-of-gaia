import numpy as np

from shared.types import TileMap


class WorldMap:
    def __init__(self, tile_map: TileMap):
        self._map = tile_map
        self._entity_layer = None
        print(self._map)
