from typing import Dict, Any

from biome.systems.maps.procedural_maps import Map
from simulation.render.tiles.manager import TerrainTileManager, TerrainSpritesMapping


class Component:
    name = None

    def __init__(self, name: str):
        self.name = name

    def update(self):
        pass

    def render(self, screen):
        pass


class MapComponent(Component):
    def __init__(self, map: Map, tile_config: Dict[str, Any]):
        super().__init__("map")
        self._map: Map = map
        self._tile_manager: TerrainTileManager = TerrainTileManager(tile_config)
        self._terrain_sprites: TerrainSpritesMapping = self._tile_manager.extract_terrain_sprites()
        self._map_surface = self._tile_manager.calculate_map(map.tile_map)

    def render(self, screen):
        for image, coords in self._map_surface:
            screen.blit(image, coords)

    @property
    def data(self):
        return self._map