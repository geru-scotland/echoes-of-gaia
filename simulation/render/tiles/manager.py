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
import logging

from typing import Dict, List, Tuple, Optional, Any, TextIO

from pygame import Surface
import pygame
import yaml

from shared.enums import TerrainType
from shared.stores.biome_store import BiomeStore
from shared.types import TileMappings, TerrainSpritesMapping, TerrainList


class TerrainTileManager:
    def __init__(self, tile_config: Dict[str, Any]):
        # TODO: Nombres de loggers, a constantes, chapuza esto.
        self._logger: logging.Logger = logging.getLogger("render_engine")
        self._tilesheet: Optional[Surface] = None
        self._tile_size: Optional[int] = None
        self._terrain_tiles_file: Optional[TextIO] = None
        self._terrain_data: Optional[TileMappings] = None
        self._load_data(tile_config)
        self._terrain_sprites: Optional[TerrainSpritesMapping] = {}

    def _load_data(self, tile_config: Dict[str, Any]):
        try:
            tilesheet_data: Dict[str, Any] = tile_config.get("assets", {}).get("tilesheets", {})
            self._tilesheet = pygame.image.load(str(tilesheet_data.get("map", {})))
            self._tile_size = tilesheet_data.get("tilesize", {})
            terrain_tiles_file: str = str(tile_config.get("configs", {}).get("terrain_files", {}))

            with open(terrain_tiles_file, "r") as file:
                self._terrain_data = yaml.safe_load(file)

        except Exception as e:
            self._logger.exception(f"There was a problem loading data within the TerrainTileManager: {e}")

    def extract_terrain_sprites(self) -> TerrainSpritesMapping:
        try:
            for terrain_type, coord_list in self._terrain_data.items():
                sprite_list: List[Surface] = []
                for coords in coord_list:
                    sprite: Surface = self._tilesheet.subsurface((coords[0], coords[1], self._tile_size, self._tile_size))
                    sprite_list.append(sprite)
                self._terrain_sprites[getattr(TerrainType, terrain_type)] = sprite_list
            return self._terrain_sprites
        except Exception as e:
            self._logger.exception(f"There was a problem extracting terrain's sprites: {e}")

    def get_tile_index_for_type(self, tile_corners: List[TerrainType], terrain_type: TerrainType) -> int:
        tile_index = 0
        for power, corner_type in enumerate(tile_corners):
            if corner_type == terrain_type:
                tile_index += 2 ** power
        return tile_index

    def calculate_map(self, terrain_type_map: List[List[TerrainType]]) -> List[Tuple[Surface, Tuple[int, int]]]:
        terrain_types: TerrainList = BiomeStore.terrains
        sprites_with_positions: List[Tuple[Surface, Tuple[int, int]]] = []
        rows = len(terrain_type_map)
        if rows < 2:
            return sprites_with_positions
        cols = len(terrain_type_map[0])
        if cols < 2:
            return sprites_with_positions

        for y in range(rows - 1):
            for x in range(cols - 1):

                tile_corner_types = [
                    terrain_type_map[y + 1][x + 1],
                    terrain_type_map[y + 1][x],
                    terrain_type_map[y][x + 1],
                    terrain_type_map[y][x]
                ]

                for terrain_type in terrain_types:
                    if terrain_type in tile_corner_types:
                        tile_index = self.get_tile_index_for_type(tile_corner_types, terrain_type)
                        image = self._terrain_sprites[terrain_type][tile_index]
                        break

                pos_x = x * self._tile_size
                pos_y = y * self._tile_size
                sprites_with_positions.append((image, (pos_x, pos_y)))

        return sprites_with_positions