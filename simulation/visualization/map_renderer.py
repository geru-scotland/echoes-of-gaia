""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""

"""
Renders terrain maps with visual styling and interaction support.

Creates visual terrain surfaces with color-coded terrain types;
handles grid display and modern visual effects with shadows.
Supports cell-based position queries and terrain information lookup -
provides optimized surface caching for performance efficiency.
"""

import logging
import random
from typing import Dict, List, Tuple, Optional, Set

import pygame
import numpy as np

from simulation.visualization.types import TerrainMapData, Color, TerrainTypeInfo, Point


class MapRenderer:
    def __init__(self, cell_size: int, grid_color: Color, terrain_colors: Dict[int, Color], show_grid: bool = True):
        self._logger = logging.getLogger("map_renderer")
        self._cell_size = cell_size
        self._grid_color = grid_color
        self._terrain_colors = terrain_colors
        self._terrain_map: Optional[np.ndarray] = None
        self._terrain_types: Dict[int, str] = {}
        self._map_dimensions: Tuple[int, int] = (0, 0)
        self._map_surface: Optional[pygame.Surface] = None
        self._show_grid = show_grid
        self._modern_style =  False
        self._cached_terrain_map: Optional[np.ndarray] = None

    def set_map_data(self, terrain_data: TerrainMapData) -> None:
        try:
            self._terrain_map = np.array(terrain_data["terrain_map"], dtype=int)
            self._terrain_types = {int(k): v for k, v in terrain_data["terrain_types"].items()}
            self._map_dimensions = tuple(terrain_data["map_dimensions"])

            if self._terrain_map.shape[0] != self._map_dimensions[0] + 1 or \
                    self._terrain_map.shape[1] != self._map_dimensions[1] + 1:
                self._logger.warning(
                    f"Map dimensions mismatch: {self._terrain_map.shape} vs expected {self._map_dimensions}"
                )

            self._update_map_surface()

            self._logger.info(f"Map data set: {self._map_dimensions}")
        except Exception as e:
            self._logger.error(f"Error setting map data: {e}")

    def _update_map_surface(self) -> None:
        if self._terrain_map is None:
            return

        width = self._terrain_map.shape[1] * self._cell_size
        height = self._terrain_map.shape[0] * self._cell_size
        self._map_surface = pygame.Surface((width, height))

        self._map_surface.fill((15, 15, 20))

        for y in range(self._terrain_map.shape[0]):
            for x in range(self._terrain_map.shape[1]):
                terrain_type = self._terrain_map[y, x]
                color = self._terrain_colors.get(terrain_type, (150, 150, 150))

                rect = pygame.Rect(
                    x * self._cell_size,
                    y * self._cell_size,
                    self._cell_size,
                    self._cell_size
                )

                inner_rect = rect.inflate(-2, -2)
                pygame.draw.rect(self._map_surface, color, inner_rect, border_radius=2)

                if self._show_grid:
                    pygame.draw.rect(self._map_surface, self._grid_color, rect, 1)

        self._cached_terrain_map = self._terrain_map.copy()

    def render(self, surface: pygame.Surface, offset: Point = (0, 0)) -> None:
        if self._map_surface is None:
            return

        if self._modern_style:
            map_rect = pygame.Rect(
                offset[0], offset[1],
                self._map_surface.get_width(),
                self._map_surface.get_height()
            )

            shadow_rect = map_rect.inflate(8, 8)
            pygame.draw.rect(surface, (20, 20, 24), shadow_rect)

            if self._cached_terrain_map is None or not np.array_equal(self._terrain_map, self._cached_terrain_map):
                self._update_map_surface()

            surface.blit(self._map_surface, offset)
        else:
            map_rect = pygame.Rect(
                offset[0], offset[1],
                self._map_surface.get_width(),
                self._map_surface.get_height()
            )
            shadow_rect = map_rect.inflate(8, 8)
            pygame.draw.rect(surface, (20, 20, 24), shadow_rect)

            surface.blit(self._map_surface, offset)

    def get_cell_at_pos(self, pos: Point) -> Optional[Tuple[int, int]]:
        if self._terrain_map is None:
            return None

        x, y = pos
        cell_x = x // self._cell_size
        cell_y = y // self._cell_size

        if 0 <= cell_y < self._terrain_map.shape[0] and 0 <= cell_x < self._terrain_map.shape[1]:
            return (cell_y, cell_x)

        return None

    def get_terrain_info(self, cell: Tuple[int, int]) -> Optional[TerrainTypeInfo]:
        if self._terrain_map is None:
            return None

        y, x = cell
        if 0 <= y < self._terrain_map.shape[0] and 0 <= x < self._terrain_map.shape[1]:
            terrain_id = self._terrain_map[y, x]
            return {
                'id': terrain_id,
                'name': self._terrain_types.get(terrain_id, f"Unknown ({terrain_id})"),
                'color': self._terrain_colors.get(terrain_id, (150, 150, 150))
            }

        return None

    def get_map_pixel_size(self) -> Tuple[int, int]:
        if self._terrain_map is None:
            return (0, 0)

        return (
            self._terrain_map.shape[1] * self._cell_size,
            self._terrain_map.shape[0] * self._cell_size
        )

    def get_cell_size(self) -> int:
        return self._cell_size