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
from typing import Dict, Any, Optional

import numpy as np

from biome.entities.fauna import Fauna
from biome.systems.behaviours.foraging import ForagingBehaviour
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.managers.worldmap_manager import WorldMapManager
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from research.training.reinforcement.reinforcement_model import ReinforcementLearningModel
from shared.enums.enums import FaunaAction, Agents, Direction, BiomeType, SimulationMode, DietType
from shared.enums.strings import Loggers
from shared.types import Observation, Position
from biome.agents.base import Agent
from utils.loggers import LoggerManager


class FaunaAgentAI(Agent[Observation, FaunaAction]):
    def __init__(self, fauna_model: str, entity_provider: EntityProvider, worldmap_manager: WorldMapManager,
                 biome_type: BiomeType):
        self._model = ReinforcementLearningModel(Agents.Reinforcement.FAUNA, fauna_model)
        self._logger = LoggerManager.get_logger(Loggers.FAUNA_AGENT)
        self._entity_provider = entity_provider
        self._worldmap_manager = worldmap_manager
        self._foraging_behaviour = ForagingBehaviour(worldmap_manager)
        self._biome_type_idx: int = list(BiomeType).index(biome_type)

    def perceive(self) -> Observation:
        fauna_entities = self._entity_provider.get_fauna(only_alive=True)
        return {
            "entities": fauna_entities
        }

    def decide(self, observation: Observation) -> Dict[int, FaunaAction]:
        actions = {}
        entities = observation["entities"]
        exclude_target_id: Optional[int] = None

        if TrainingTargetManager.get_training_mode() == SimulationMode.TRAINING_WITH_RL_MODEL:
            exclude_target_id = TrainingTargetManager.get_target_id()

        for entity in entities:
            if exclude_target_id is not None and entity.get_id() == exclude_target_id:
                continue

            entity_observation = self._prepare_entity_observation(entity)
            action = self._model.predict(entity_observation)
            actions[entity.get_id()] = list(FaunaAction).index(action)

        self._logger.debug(f"ACTIONS: {actions}")
        return actions

    def act(self, actions: Dict[int, FaunaAction]) -> None:
        self._logger.debug("Going to act:")

        exclude_target_id: Optional[int] = None

        if TrainingTargetManager.get_training_mode() == SimulationMode.TRAINING_WITH_RL_MODEL:
            exclude_target_id = TrainingTargetManager.get_target_id()

        for entity_id, action in actions.items():
            try:
                if exclude_target_id is not None and entity_id == exclude_target_id:
                    continue

                fauna_entity: Fauna = self._worldmap_manager.get_entity_by_id(entity_id)
                if not fauna_entity or not fauna_entity.is_alive():
                    continue

                self._foraging_behaviour.set_target(fauna_entity)

                if fauna_entity.energy_reserves > 0:
                    self._logger.debug(f"Trying to move entity {entity_id} to {action}")
                    direction = self._action_to_direction(action)
                    self._logger.debug(f"Moving to {direction}")
                    self._logger.debug(f"Previous positin {fauna_entity.get_position()}")
                    previous_position: Position = fauna_entity.get_position()
                    fauna_entity.move(direction)
                    new_position: Position = fauna_entity.get_position()

                    if previous_position != new_position and self._foraging_behaviour:
                        self._foraging_behaviour.check_and_drink_water()
                        self._foraging_behaviour.check_and_eat_food()

                    self._logger.debug(f"Post position {fauna_entity.get_position()}")

            except Exception as e:
                self._logger.warning(f"Error executing action for fauna entity {entity_id}: {e}")

    def _prepare_entity_observation(self, entity: Fauna) -> Dict[str, Any]:
        position = entity.get_position()
        fov_size = 15

        terrain_map = np.zeros((fov_size, fov_size), dtype=np.int32)
        validity_map = np.zeros((fov_size, fov_size), dtype=np.float32)
        visited_map = np.zeros((fov_size, fov_size), dtype=np.float32)
        flora_map = np.zeros((fov_size, fov_size), dtype=np.int8)
        prey_map = np.zeros((fov_size, fov_size), dtype=np.int8)
        predator_map = np.zeros((fov_size, fov_size), dtype=np.int8)

        if position:
            local_maps = self._worldmap_manager.get_local_maps(position, entity.diet_type, entity.get_species(),
                                                               fov_size, fov_size)
            if local_maps:
                terrain_map, validity_map, flora_map, prey_map, predator_map = local_maps

                terrain_map = terrain_map.astype(np.int64)
                validity_map = validity_map.astype(np.float32)
                flora_map = flora_map.astype(np.float32)
                prey_map = prey_map.astype(np.float32)
                predator_map = predator_map.astype(np.float32)

                visited_mask = self._worldmap_manager.get_local_visited_map(entity, fov_size, fov_size)
                visited_map = visited_mask.astype(np.float32)

        thirst_level = 0.0
        energy_reserves = 0.00
        vitality = 0.0
        stress_level = 0.0
        hunger_level = 0.0
        somatic_integrity = 0.0
        diet_type_idx = 0

        if entity and entity.is_alive():
            thirst_level = entity.thirst_level / 100.0  # OJO, normalizo, que si no no tiraba bien.
            energy_reserves = entity.energy_reserves / entity.max_energy_reserves

            vitality = entity.vitality / entity.max_vitality
            stress_level = entity.stress_level / 100.0
            hunger_level = entity.hunger_level / 100.0
            somatic_integrity = entity.somatic_integrity / entity.max_somatic_integrity
            diet_type_idx = list(DietType).index(entity.diet_type)

        return {
            "biome_type": self._biome_type_idx,
            "diet_type": diet_type_idx,
            "terrain_map": terrain_map,
            "validity_map": validity_map,
            "visited_map": visited_map,
            "flora_map": flora_map,
            "prey_map": prey_map,
            "predator_map": predator_map,
            "thirst_level": np.array([thirst_level], dtype=np.float32),
            "hunger_level": np.array([hunger_level], dtype=np.float32),
            "energy_reserves": np.array([energy_reserves], dtype=np.float32),
            "vitality": np.array([vitality], dtype=np.float32),
            "stress_level": np.array([stress_level], dtype=np.float32),
            "somatic_integrity": np.array([somatic_integrity], dtype=np.float32),
        }

    def _action_to_direction(self, action: FaunaAction) -> Direction:
        action_map = {
            FaunaAction.MOVE_NORTH: Direction.NORTH,
            FaunaAction.MOVE_SOUTH: Direction.SOUTH,
            FaunaAction.MOVE_EAST: Direction.EAST,
            FaunaAction.MOVE_WEST: Direction.WEST,
        }
        return action_map.get(action)
