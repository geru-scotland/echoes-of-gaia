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
Entity interaction system for simulating predator-prey dynamics in biome training.

Manages spatial interactions between fauna entities during training; processes
attack and defense scenarios based on diet types and proximity. Tracks interaction
history and calculates movement vectors relative to threats - provides directional
awareness for entities navigating complex biome environments.
"""

import random
from logging import Logger
from typing import List, Optional, Tuple

from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.systems.managers.worldmap_manager import WorldMapManager
from shared.behaviour_types import (
    EntityID,
    InteractionResult,
    Position,
)
from shared.enums.enums import DietType, Direction, EntityType
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class LocalInteractionSimulator:
    def __init__(self, worldmap_manager: WorldMapManager):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._worldmap: WorldMapManager = worldmap_manager
        self._target_entity: Optional[Fauna] = None
        self._interaction_records: List[Tuple[EntityID, EntityID, InteractionResult]] = []

    def set_target(self, target: Fauna) -> None:
        self._target_entity = target

    def process_interactions(self) -> None:
        if not self._target_entity or not self._target_entity.is_alive():
            return

        target_position = self._target_entity.get_position()
        if not target_position:
            return

        nearby_entities = self._worldmap.get_entities_near(target_position, radius=1).get(EntityType.FAUNA, [])

        entities_at_position = [e for e in nearby_entities if
                                e.is_alive() and e.get_id() != self._target_entity.get_id()
                                and e.get_species() != self._target_entity.get_species()]

        if entities_at_position and self._target_entity.diet_type == DietType.HERBIVORE:
            self._process_prey_interactions(self._target_entity, entities_at_position)
            # if self._target_entity.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE]:
            #     # Target es depredador: puede atacar a presas
            #     self._process_predator_interactions(self._target_entity, entities_at_position)

    def _process_predator_interactions(self, predator: Fauna, entities: List[Entity]) -> None:
        potential_prey = [e for e in entities
                          if isinstance(e, Fauna)
                          and e.diet_type == DietType.HERBIVORE
                          and e.is_alive()
                          and e.get_species() != predator.get_species()]

        for prey in potential_prey:
            damage = 15.0 + random.uniform(-5.0, 5.0)
            prey.apply_damage(damage, predator.get_id())
            self._logger.info(
                f"Agent target got damaged for {damage} points by predator: {predator.get_id()} ({predator.get_species()})")

            result: InteractionResult = "attacked"

            if not prey.is_alive():
                result = "killed"
                self._logger.info(f"Target predator {predator.get_id()} killed prey {prey.get_id()}")
            else:
                self._logger.info(f"Target predator {predator.get_id()} attacked prey {prey.get_id()}")

            self._interaction_records.append((predator.get_id(), prey.get_id(), result))

    def _process_prey_interactions(self, prey: Fauna, entities: List[Entity]) -> None:
        potential_predators = [e for e in entities
                               if isinstance(e, Fauna)
                               and e.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE]
                               and e.is_alive()
                               and e.get_species() != prey.get_species()]

        for predator in potential_predators:
            damage = 12.0 + random.uniform(-4.0, 4.0)
            prey.apply_damage(damage, predator.get_id())

            result: InteractionResult = "attacked"

            if not prey.is_alive():
                result = "killed"
                self._logger.info(f"Predator {predator.get_id()} killed target prey {prey.get_id()}")
            else:
                self._logger.info(f"Predator {predator.get_id()} attacked target prey {prey.get_id()}")

            self._interaction_records.append((predator.get_id(), prey.get_id(), result))

    def get_recent_interactions(self, limit: int = 10) -> List[Tuple[EntityID, EntityID, InteractionResult]]:
        return self._interaction_records[-limit:] if self._interaction_records else []

    def is_moving_away_from(self, my_pos: Position, other_pos: Position, direction: Direction) -> bool:
        if direction is None or not my_pos or not other_pos:
            return False

        vec_to_other = (other_pos[0] - my_pos[0], other_pos[1] - my_pos[1])

        dot_product = vec_to_other[0] * direction.value[0] + vec_to_other[1] * direction.value[1]
        return dot_product < 0

    def is_moving_toward(self, my_pos: Position, other_pos: Position, direction: Direction) -> bool:
        return not self.is_moving_away_from(my_pos, other_pos, direction)

    def calculate_manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        if not pos1 or not pos2:
            return float('inf')
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
