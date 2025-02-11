import logging

from typing import Dict, List, Tuple, Optional, Any, TextIO

from pygame import Surface
import pygame
import yaml
from shared.enums import TerrainType

# Tipos, me acabo de enterar que puedo definir tipos custom en Python,
# algo parecido al typedef de c++; y soy un poco más feliz:
Coords = Tuple[int, int]
TileMappings = Dict[TerrainType, List[Coords]]
TerrainSpritesMapping = Dict[TerrainType, List[Surface]]

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

