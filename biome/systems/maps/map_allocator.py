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
Spatial management system for entity position allocation.

Handles entity placement and movement in the terrain grid;
manages position validation and habitat-based space allocation.
Tracks entity positions with optimized index map implementation.
"""

from logging import Logger
from typing import List, Optional, Tuple

import numpy as np

from biome.systems.managers.habitat_manager import HabitatManager
from shared.enums.strings import Loggers
from shared.types import EntityIndexMap, Position, TileMap
from utils.loggers import LoggerManager


class MapAllocator:
    def __init__(self, tile_map: TileMap):
        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._tile_map: TileMap = tile_map
        self._entity_index_map: EntityIndexMap = np.full(tile_map.shape, -1, dtype=int)
        self._habitat_manager = HabitatManager(tile_map)

    def is_position_valid(self, position: Position) -> bool:
        if position is None:
            return False

        y, x = position
        height, width = self._entity_index_map.shape

        return 0 <= y < height and 0 <= x < width

    def allocate_position(self, entity_id: int, habitats: List) -> Optional[Position]:
        position = self._habitat_manager.get_available_position(habitats)

        if position is None:
            self._logger.warning(
                f"No available position found for entity {entity_id} with habitats {habitats}")
            return None

        if self._entity_index_map[position] != -1:
            self._logger.warning(
                f"Position {position} is already occupied in index map by entity {self._entity_index_map[position]}")
            return None

        self._habitat_manager.remove_from_available_habitats(position)

        self._entity_index_map[position] = entity_id

        return position

    def free_position(self, position: Position, habitats: List) -> int:
        if position is None:
            return -1

        entity_id = self._entity_index_map[position]

        if entity_id == -1:
            return -1

        self._entity_index_map[position] = -1

        self._habitat_manager.restore_habitat(position)

        return entity_id

    def move_entity(self, entity_id: int, old_position: Position, new_position: Position, habitats: List) -> bool:
        if old_position is None or new_position is None:
            return False

        if not self.is_position_valid(new_position):
            self._logger.debug(
                f"Target position {new_position} is outside map boundaries")
            return False

        if self._entity_index_map[old_position] != entity_id:
            self._logger.debug(
                f"Entity {entity_id} is not at position {old_position}")
            return False

        if self._entity_index_map[new_position] != -1:
            self._logger.debug(
                f"Target position {new_position} is already occupied by entity {self._entity_index_map[new_position]}")
            return False

        self._entity_index_map[old_position] = -1
        self._entity_index_map[new_position] = entity_id

        # Pongo nombres más expresivos, que me hice la picha un lio.
        self._habitat_manager.restore_habitat(old_position)
        self._habitat_manager.remove_from_available_habitats(new_position)

        return True

    def is_position_occupied(self, position: Position) -> bool:
        return self._entity_index_map[position] != -1

    def get_entity_at(self, position: Position) -> int:
        if position is None or not (0 <= position[0] < self._entity_index_map.shape[0] and
                                    0 <= position[1] < self._entity_index_map.shape[1]):
            return -1
        return self._entity_index_map[position]

    def get_map_shape(self) -> Tuple[int, int]:
        return self._entity_index_map.shape

    @property
    def entity_index_map(self) -> EntityIndexMap:
        return self._entity_index_map
