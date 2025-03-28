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

import numpy as np

from biome.biome import Biome
from biome.entities.entity import Entity
from config.settings import Settings
from research.training.reinforcement.adapter import EnvironmentAdapter
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import ComponentType, EntityType, SimulationMode, FaunaSpecies
from shared.enums.events import SimulationEvent
from shared.enums.strings import Loggers
from shared.normalization.normalizer import climate_normalizer
from simulation.api.simulation_api import SimulationAPI
from simulation.core.systems.events.event_bus import SimulationEventBus
from utils.loggers import LoggerManager


class FaunaSimulationAdapter(EnvironmentAdapter):
    def __init__(self):
        self._logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._simulation_api: SimulationAPI = None
        self._biome: Biome = None
        self._target: Entity = None
        self._logger.info(f"Creating {self.__class__.__name__}...")
        self._register_events()

    def _register_events(self):
        SimulationEventBus.register(SimulationEvent.SIMULATION_TRAIN_TARGET_ACQUIRED, self._handle_target_acquired)

    def _handle_target_acquired(self, *args, **kwargs):
        entity: Entity = kwargs.get("entity", None)
        generation: int = kwargs.get("generation", None)

        if entity and generation:
            self._target = entity
            TrainingTargetManager.mark_as_acquired()

            self._logger.info(
                f"Training target acquired: {entity.get_species()} (ID: {entity.get_id()}) - Generation {generation}")

    def initialize(self):
        self._logger.info(f"Initializing {self.__class__.__name__} ...")

        settings = Settings(override_configs="training.yaml")

        self._simulation_api = SimulationAPI(
            settings=settings,
            mode=SimulationMode.TRAINING,
        )

        self._simulation_api.initialise_training()
        self._biome = self._simulation_api.get_biome()

        TrainingTargetManager.set_training_mode(True)
        TrainingTargetManager.reset()

        target_species = self._select_target_species()
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

        return selected_species

    def _wait_for_target_acquisition(self):
        max_attempts = 5000
        attempts = 0

        self._logger.info("Waiting for target acquisition...")

        while not TrainingTargetManager.is_acquired() and attempts < max_attempts:
            self._simulation_api.step(1)
            attempts += 1

            if attempts % 100 == 0:
                self._logger.debug(f"Waiting for target... {attempts} steps")

        if not TrainingTargetManager.is_acquired():
            raise RuntimeError("Failed to acquire target after maximum attempts")

        self._logger.info("Target acquired successfully")

    def compute_reward(self, action):
        if not self._target or not self._target.is_alive():
            return -10.0

        vital_component = self._target.get_component(ComponentType.VITAL)
        if not vital_component:
            return 0.0

        vitality_ratio = vital_component.vitality / vital_component.max_vitality
        reward = vitality_ratio * 2.0

        return reward

    def get_observation(self):
        if not self._target:
            return self._get_default_observation()

        observation = {}

        # 1. Estado interno de la fauna
        vital_component = self._target.get_component(ComponentType.VITAL)
        if vital_component:
            observation["vitality"] = np.array([vital_component.vitality / vital_component.max_vitality],
                                               dtype=np.float32)
            observation["age"] = np.array([vital_component.age / vital_component.lifespan],
                                          dtype=np.float32)

        # 2. Información sobre el entorno cercano
        position = self._target.get_position()
        if position:
            nearby_entities = self._find_nearby_entities(position, radius=3)
            observation["nearby_flora"] = np.array([len([e for e in nearby_entities
                                                         if e.get_type() == EntityType.FLORA])],
                                                   dtype=np.float32)
            observation["nearby_fauna"] = np.array([len([e for e in nearby_entities
                                                         if e.get_type() == EntityType.FAUNA])],
                                                   dtype=np.float32)

        # 3. Información climática
        climate_system = self._biome._climate
        if climate_system:
            climate_state = climate_system.get_state()
            observation["temperature"] = np.array([climate_normalizer.normalize(
                "temperature", climate_state.temperature)],
                dtype=np.float32)
            observation["humidity"] = np.array([climate_normalizer.normalize(
                "humidity", climate_state.humidity)],
                dtype=np.float32)

        return observation

    def _find_nearby_entities(self, position, radius):
        return []

    def _get_default_observation(self):
        return {}

    def step_environment(self, action: int, time_delta: int = 1) -> None:
        if not self._simulation_api or not self._target:
            return

        self._simulation_api.step(time_delta)

    def finish_training(self):
        if self._simulation_api:
            self._simulation_api.finish_training()

        TrainingTargetManager.set_training_mode(False)

    def get_target(self) -> Entity:
        return self._target
