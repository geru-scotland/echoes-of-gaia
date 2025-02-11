from typing import Optional, List, Dict, Any

import pygame
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
    def __init__(self, map: Map, width: int, height: int, tile_config: Dict[str, Any]):
        super().__init__("map")
        self._map: Map = map
        self.map_surface = pygame.Surface((width, height))
        self._tile_manager: TerrainTileManager = TerrainTileManager(tile_config)
        self._terrain_sprites: TerrainSpritesMapping = self._tile_manager.extract_terrain_sprites()
    def render(self, screen):
        screen.blit(self.map_surface, (0, 0))

    @property
    def data(self):
        return self._map