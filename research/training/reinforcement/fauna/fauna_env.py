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
from shared.enums.enums import FaunaAction, Agents, LocalFovConfig, TerrainType
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


@EnvironmentRegistry.register(Agents.Reinforcement.FAUNA)
class FaunaEnvironment(gym.Env):

    def __init__(self, local_fov_config: LocalFovConfig):
        super().__init__()

        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self.action_space: Discrete = gym.spaces.Discrete(len(FaunaAction))

        self._fov_width: int = local_fov_config.get("size", {}).get("width", 10)
        self._fov_height: int = local_fov_config.get("size", {}).get("height", 10)
        self._fov_center: int = local_fov_config.get("center", int(self._fov_width / 2))

        self.observation_space = spaces.Dict({
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
            )
        })

        self._current_step: int = 0
        # Pongo por ahora cap, por si en algún momento no mueren.
        self._max_episode_steps: int = 8000
        self._finished: bool = False
        self._fauna_adapter: FaunaSimulationAdapter = None

        atexit.register(self.cleanup)

    def cleanup(self):
        if self._fauna_adapter:
            self._fauna_adapter.finish_training_session()

    def reset(self, *, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)

        observation = self._get_default_observation()

        if not self._finished:
            self._fauna_adapter = FaunaSimulationAdapter(self._fov_width, self._fov_height, self._fov_center)
            self._fauna_adapter.initialize()

            observation = self._fauna_adapter.get_observation()

        return observation, {}

    def step(self, action):
        self._current_step += 1

        self._logger.info(f"Received action: {action}")
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

        return {
            "terrain_map": terrain_map,
            "validity_map": valid_mask,
            "visited_map": visited_mask,
        }
