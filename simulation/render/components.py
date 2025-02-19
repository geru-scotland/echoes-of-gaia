"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from typing import Dict, Any

from biome.systems.maps.procedural_maps import MapGenData
from shared.types import TileMap
from simulation.render.tiles.manager import TerrainTileManager, TerrainSpritesMapping


class RenderComponent:
    name = None

    def __init__(self, name: str):
        self.name = name

    def update(self):
        pass

    def render(self, screen):
        pass


class MapComponent(RenderComponent):
    def __init__(self, tile_map: TileMap, tile_config: Dict[str, Any]):
        super().__init__("map")
        self._tile_map: TileMap= tile_map
        self._tile_manager: TerrainTileManager = TerrainTileManager(tile_config)
        self._terrain_sprites: TerrainSpritesMapping = self._tile_manager.extract_terrain_sprites()
        self._map_surface = self._tile_manager.calculate_map(self._tile_map)

    def render(self, screen):
        for image, coords in self._map_surface:
            screen.blit(image, coords)

    @property
    def data(self):
        return self._tile_map