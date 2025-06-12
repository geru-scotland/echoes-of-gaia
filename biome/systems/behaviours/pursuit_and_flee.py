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
Predator-prey movement system for fauna entities.

Manages pursuit and fleeing behaviors based on diet types; calculates
optimal movement directions for predators chasing prey and prey escaping.
Tracks entity states dynamically and modifies movement patterns according
to target entity properties - implements ecological chase dynamics.
"""

from logging import Logger
from typing import Dict, List, Tuple, Optional

from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.systems.managers.worldmap_manager import WorldMapManager
from shared.behaviour_types import EntityID, EntityState
from shared.enums.enums import EntityType, Direction, DietType
from shared.types import Position
from utils.loggers import LoggerManager
from shared.enums.strings import Loggers


class PursuitAndFleeBehaviour:

    def __init__(self, worldmap_manager: WorldMapManager):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._worldmap: WorldMapManager = worldmap_manager
        self._entity_states: Dict[EntityID, EntityState] = {}
        self._target_entity: Optional[Fauna] = None

    def set_target(self, target: Fauna) -> None:
        self._target_entity = target

    def process_entity_behaviours(self, entities: List[Entity]) -> None:
        if not self._target_entity or not self._target_entity.is_alive():
            return

        target_id = self._target_entity.get_id()
        target_position = self._target_entity.get_position()
        target_species = self._target_entity.get_species()
        target_diet_type = self._target_entity.diet_type

        if not target_position:
            return

        for entity in entities:
            if not isinstance(entity, Fauna) or not entity.is_alive():
                continue

            entity_id = entity.get_id()

            if entity_id == target_id:
                continue

            if entity.get_species() == target_species:
                self._entity_states[entity_id] = "inactive"
                continue

            if target_diet_type in [DietType.CARNIVORE, DietType.OMNIVORE] and entity.diet_type == DietType.HERBIVORE:
                self._entity_states[entity_id] = "fleeing"
                self._execute_flee_behaviour(entity, target_position)

            elif target_diet_type == DietType.HERBIVORE and entity.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE]:
                self._entity_states[entity_id] = "hunting"
                self._execute_hunt_behaviour(entity, target_position)

            else:
                self._entity_states[entity_id] = "inactive"

    def _execute_flee_behaviour(self, entity: Fauna, target_position: Position) -> None:
        my_pos = entity.get_position()
        if not my_pos:
            return

        flee_direction = self._calculate_escape_direction(my_pos, target_position)

        entity.move(flee_direction)
        self._logger.debug(f"Entity {entity.get_id()} fleeing from target in direction {flee_direction}")

    def _execute_hunt_behaviour(self, entity: Fauna, target_position: Position) -> None:
        my_pos = entity.get_position()
        if not my_pos:
            return

        hunt_direction = self._calculate_pursuit_direction(my_pos, target_position)

        entity.move(hunt_direction)
        self._logger.debug(f"Entity {entity.get_id()} hunting target in direction {hunt_direction}")

    def _calculate_manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _calculate_escape_direction(self, my_pos: Position, threat_pos: Position) -> Direction:
        y_diff = my_pos[0] - threat_pos[0]
        x_diff = my_pos[1] - threat_pos[1]

        if abs(y_diff) > abs(x_diff):
            return Direction.NORTH if y_diff < 0 else Direction.SOUTH
        else:
            return Direction.WEST if x_diff < 0 else Direction.EAST

    def _calculate_pursuit_direction(self, my_pos: Position, target_pos: Position) -> Direction:
        y_diff = target_pos[0] - my_pos[0]
        x_diff = target_pos[1] - my_pos[1]

        if abs(y_diff) > abs(x_diff):
            return Direction.NORTH if y_diff < 0 else Direction.SOUTH
        else:
            return Direction.WEST if x_diff < 0 else Direction.EAST
