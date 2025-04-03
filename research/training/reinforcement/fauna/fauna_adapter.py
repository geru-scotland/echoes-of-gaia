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
import random
from logging import Logger
from typing import Tuple, Optional

import numpy as np

from biome.biome import Biome
from biome.entities.entity import Entity
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.managers.worldmap_manager import WorldMapManager
from config.settings import Settings
from research.training.reinforcement.adapter import EnvironmentAdapter
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import ComponentType, EntityType, SimulationMode, FaunaSpecies, Direction, FaunaAction, \
    PositionNotValidReason, TerrainType
from shared.enums.events import SimulationEvent
from shared.enums.strings import Loggers
from shared.types import Position, DecodedAction
from simulation.api.simulation_api import SimulationAPI
from simulation.core.systems.events.event_bus import SimulationEventBus
from utils.loggers import LoggerManager


class FaunaSimulationAdapter(EnvironmentAdapter):
    def __init__(self, fov_width: int, fov_height: int, fov_center):
        self._logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._simulation_api: SimulationAPI = None
        self._biome: Biome = None
        self._worldmap_manager: WorldMapManager = None
        self._target: Entity = None
        self._logger.info(f"Creating {self.__class__.__name__}...")
        self._register_events()
        self._fov_width: int = fov_width
        self._fov_height: int = fov_height
        self._fov_center: int = fov_center
        self._visited_positions = set()
        self._heatmap = {}

    def __del__(self):
        try:
            self._logger.info("Closing Fauna Adapter resources...")
            self.finish_training_session()
        except Exception as e:
            self._logger.warning(f"Error during cleanup: {e}")

    def _register_events(self):
        SimulationEventBus.register(SimulationEvent.SIMULATION_TRAIN_TARGET_ACQUIRED, self._handle_target_acquired)

    def _handle_target_acquired(self, *args, **kwargs):
        entity: Entity = kwargs.get("entity", None)
        generation: int = kwargs.get("generation", None)

        if entity and generation and not TrainingTargetManager.is_acquired():
            self._target = entity
            TrainingTargetManager.mark_as_acquired()
            self._logger.info(
                f"Training target acquired: {entity.get_species()} (ID: {entity.get_id()}) - Generation {generation}")

    def _is_new_position(self, position: Position) -> bool:
        return position not in self._visited_positions

    def initialize(self):
        self._logger.info(f"Initializing {self.__class__.__name__} ...")

        settings = Settings(override_configs="training.yaml")

        SimulationEventBus.clear()
        BiomeEventBus.clear()

        self._register_events()
        self._simulation_api = SimulationAPI(
            settings=settings,
            mode=SimulationMode.TRAINING,
        )

        self._simulation_api.initialise_training()
        self._biome = self._simulation_api.get_biome()
        self._worldmap_manager = self._biome.get_worldmap_manager()

        TrainingTargetManager.reset()
        TrainingTargetManager.set_training_mode(True)

        target_species = self._select_target_species()
        # TODO: OJO! Que no se te olvide poner esto debidamente, la idea es
        # probar diversos rangos de generaciones.
        target_generation = random.randint(1, 4)

        TrainingTargetManager.set_target(
            str(target_species),
            EntityType.FAUNA,
            target_generation
        )

        self._logger.info(f"Waiting for target species: {target_species}, generation: {target_generation}")

        try:
            self._wait_for_target_acquisition()
        except Exception as e:
            self._logger.warning(e)

    def _select_target_species(self) -> FaunaSpecies:
        available_species = list(FaunaSpecies)
        selected_species = random.choice(available_species)

        return FaunaSpecies.DEER

    def _wait_for_target_acquisition(self):
        max_attempts = 10000
        attempts = 0

        self._logger.info("Waiting for target acquisition...")

        while not TrainingTargetManager.is_acquired() and attempts < max_attempts:
            self._simulation_api.step(1)
            attempts += 1

            if attempts % 100 == 0:
                self._logger.debug(f"Waiting for target... {attempts} steps")

        if not TrainingTargetManager.is_acquired():
            raise RuntimeError("Failed to acquire target after maximum attempts")

        self._logger.info(f"Target acquired successfully: {TrainingTargetManager.get_current_episode()}")

    def step_environment(self, action: int, time_delta: int = 1) -> None:
        if not self._simulation_api or not self._target:
            return

        old_position = self._target.get_position()

        self._target.move(self._action_decode(action))

        new_position = self._target.get_position()

        if new_position and new_position != old_position:
            self._visited_positions.add(new_position)

        self._simulation_api.step(time_delta)

    def compute_reward(self, action: FaunaAction):
        if not self._target or not self._target.is_alive():
            return -10.0

        reward: float = 0.0

        decoded_action: DecodedAction = self._action_decode(action)

        movement_reward: int = self.compute_movement_reward(decoded_action)
        water_reward: int = self.compute_water_reward()
        return reward + movement_reward

    def _action_decode(self, action: int) -> DecodedAction:
        if action < len(Direction):
            return list(Direction)[action]

    def compute_movement_reward(self, direction: Direction) -> float:
        new_position: Position = self._target.movement_component.calculate_new_position(direction)

        is_valid, reason = self._worldmap_manager.is_valid_position(new_position, self._target.get_id())

        if not is_valid:
            if reason == PositionNotValidReason.POSITION_OUT_OF_BOUNDARIES:
                return -1.0
            elif reason == PositionNotValidReason.POSITION_NON_TRAVERSABLE:
                return -0.8
            elif reason == PositionNotValidReason.POSITION_BUSY:
                return -0.6

        # Reward base por movimiento válido
        reward = 0.0

        # Bonus por exploración si la posición es nueva
        # Quiero incentivar un poco, al menos por ahora, a que explore
        if self._is_new_position(new_position):
            reward += 0.7

        return reward

    def get_observation(self):
        if not self._target:
            return self._get_default_observation()

        position = self._target.get_position()

        # Obtener mapa local y máscara de validez
        local_result = None
        if position:
            local_result = self._worldmap_manager.get_local_map(position, self._fov_width, self._fov_height)

        if local_result is None:
            # Valores por defecto si no hay información válida
            local_fov_terrain = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
            validity_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
            visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
        else:
            local_fov_terrain, validity_mask = local_result

            local_fov_terrain = local_fov_terrain.astype(np.int64)
            visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)

            center_y, center_x = self._fov_height // 2, self._fov_width // 2

            for y in range(self._fov_height):
                for x in range(self._fov_width):
                    # Convierto coords de esta celda, a posición global
                    global_y = position[0] + (y - center_y)
                    global_x = position[1] + (x - center_x)
                    global_pos = (global_y, global_x)

                    if global_pos in self._visited_positions:
                        visited_mask[y, x] = 1.0

        validity_map = validity_mask.astype(np.float32)
        visited_map = visited_mask.astype(np.float32)

        return {
            "terrain_map": local_fov_terrain,
            "validity_map": validity_map,
            "visited_map": visited_map
        }

    def _find_nearby_entities(self, position, radius):
        return []

    def _get_default_observation(self):
        terrain_map = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
        valid_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)

        return {
            "terrain_map": terrain_map,
            "validity_map": valid_mask,
            "visited_map": visited_mask,
        }

    def finish_training_session(self):
        if self._simulation_api:
            self._simulation_api.finish_training()

        TrainingTargetManager.set_training_mode(False)

    def get_target(self) -> Entity:
        return self._target
