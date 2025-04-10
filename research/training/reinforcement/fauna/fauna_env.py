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
import atexit
from logging import Logger

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Discrete

from biome.entities.entity import Entity
from research.training.registry import EnvironmentRegistry
from research.training.reinforcement.fauna.fauna_adapter import FaunaSimulationAdapter
from shared.enums.enums import FaunaAction, Agents, LocalFovConfig, TerrainType, BiomeType, DietType
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


@EnvironmentRegistry.register(Agents.Reinforcement.FAUNA)
class FaunaEnvironment(gym.Env):

    def __init__(self, local_fov_config: LocalFovConfig):
        super().__init__()
        self._episode_number: int = 0

        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self.action_space: Discrete = gym.spaces.Discrete(len(FaunaAction))

        self._fov_width: int = local_fov_config.get("size", {}).get("width", 10)
        self._fov_height: int = local_fov_config.get("size", {}).get("height", 10)
        self._fov_center: int = local_fov_config.get("center", int(self._fov_width / 2))

        self.observation_space = spaces.Dict({
            "biome_type": spaces.Discrete(len(BiomeType)),
            "diet_type": spaces.Discrete(len(DietType)),
            "terrain_map": spaces.Box(
                low=0,
                high=len(list(TerrainType)) - 1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int64
            ),
            "validity_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._fov_height, self._fov_width),
                dtype=np.float32
            ),
            "visited_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._fov_height, self._fov_width),
                dtype=np.float32
            ),
            "flora_map": spaces.Box(
                low=0,
                high=1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int8
            ),
            "prey_map": spaces.Box(
                low=0,
                high=1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int8
            ),
            "predator_map": spaces.Box(
                low=0,
                high=1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int8
            ),
            "water_map": spaces.Box(
                low=0,
                high=1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int8
            ),
            "food_map": spaces.Box(
                low=0,
                high=1,
                shape=(self._fov_height, self._fov_width),
                dtype=np.int8
            ),
            "thirst_level": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "energy_reserves": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "vitality": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "stress_level": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "hunger_level": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            ),
            "somatic_integrity": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })

        self._current_step: int = 0
        self._episode_step: int = 0
        # Pongo por ahora cap, por si en algún momento no mueren.
        self._max_episode_steps: int = 4000
        self._finished: bool = False
        self._fauna_adapter: FaunaSimulationAdapter = None

        atexit.register(self.cleanup)

    def cleanup(self):
        if self._fauna_adapter:
            self._fauna_adapter.finish_training_session()

    def reset(self, *, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)
        self._episode_step = 0
        self._episode_number += 1

        def format_boxed_title(title: str, width: int = 60, border: str = "#") -> str:
            line = border * width
            padded_title = f"{border}{title.center(width - 2)}{border}"
            return "\n" + line + "\n" + padded_title + "\n" + line + "\n"

        self._logger.info(format_boxed_title(f"STARTING EPISODE #{self._episode_number}"))

        observation = self._get_default_observation()

        max_attempts = 8
        attempt = 0
        while not self._finished and attempt < max_attempts:
            self._fauna_adapter = FaunaSimulationAdapter(self._fov_width, self._fov_height, self._fov_center)
            correct_init: bool = self._fauna_adapter.initialize()
            if correct_init:
                observation = self._fauna_adapter.get_observation()
                break
            attempt += 1
            self._logger.warning("Couldn't acquire a target. Retrying...")
        else:
            self._logger.error(f"Target acquisition failed {max_attempts} times. Aborting")

        return observation, {}

    def step(self, action):
        self._episode_step += 1
        self._current_step += 1

        if self._current_step % 250 == 0:
            line = "-" * 60
            step_msg = f"STEP CHECKPOINT: {self._episode_step} steps into EPISODE #{self._episode_number}"
            self._logger.info("\n" + line + f"\n{step_msg.center(60)}\n" + line + "\n")

        self._logger.debug(f"Received action: {action}")
        self._fauna_adapter.step_environment(action)

        info = {}
        terminated = False
        truncated = False

        observation = self._fauna_adapter.get_observation()
        target: Entity = self._fauna_adapter.get_target()

        if not target.is_alive():
            self._logger.error("TERMINATING EPISODE!!")
            self._fauna_adapter.finish_training_session()
            terminated = True

        reward = self._fauna_adapter.compute_reward(action)

        return observation, reward, terminated, truncated, info

    def _get_default_observation(self):
        terrain_map = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
        valid_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        flora_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        prey_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        predator_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        water_map = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        food_map = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)

        return {
            "biome_type": list(BiomeType).index(BiomeType.TROPICAL),
            "diet_type": list(DietType).index(DietType.HERBIVORE),
            "terrain_map": terrain_map,
            "validity_map": valid_mask,
            "visited_map": visited_mask,
            "flora_map": flora_mask,
            "prey_map": prey_mask,
            "predator_map": predator_mask,
            "water_map": water_map,
            "food_map": food_map,
            "thirst_level": np.array([1.0], dtype=np.float32),
            "energy_reserves": np.array([1.0], dtype=np.float32),
            "vitality": np.array([1.0], dtype=np.float32),
            "stress_level": np.array([0.0], dtype=np.float32),
            "hunger_level": np.array([0.0], dtype=np.float32),
            "somatic_integrity": np.array([1.0], dtype=np.float32),
        }
