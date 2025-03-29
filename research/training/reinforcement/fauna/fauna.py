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
from logging import Logger
from typing import Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete

from biome.entities.entity import Entity
from research.training.registry import EnvironmentRegistry
from research.training.reinforcement.fauna.fauna_adapter import FaunaSimulationAdapter
from shared.enums.enums import FaunaAction, Agents
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


@EnvironmentRegistry.register(Agents.Reinforcement.FAUNA)
class FaunaEnvironment(gym.Env):

    def __init__(self):
        super().__init__()

        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self.action_space: Discrete = gym.spaces.Discrete(len(FaunaAction))

        self.observation_space = spaces.Dict({
            "vitality": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "age": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "nearby_flora": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "nearby_fauna": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "temperature": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "humidity": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self._current_step: int = 0
        # Pongo por ahora cap, por si en algún momento no mueren.
        self._max_episode_steps: int = 8000
        self._finished: bool = False
        self._fauna_adapter: FaunaSimulationAdapter = None

    def reset(self, *, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)

        observation = self._get_default_observation()

        if not self._finished:
            self._fauna_adapter = FaunaSimulationAdapter()
            self._fauna_adapter.initialize()

            observation = self._fauna_adapter.get_observation()

        return observation, {}

    def step(self, action):
        self._current_step += 1
        self._fauna_adapter.step_environment(action)

        info = {}
        terminated = False
        truncated = False

        observation = self._fauna_adapter.get_observation()
        target: Entity = self._fauna_adapter.get_target()

        if not target.is_alive():
            self._logger.error("TERMINATING EPISODE!!")
            self._fauna_adapter.finish_training()
            terminated = True
            # terminated = not target.is_alive()

        self._logger.info(f"OBSERVATION FOR {target.get_species()}. Observation: {observation}")

        reward = self._fauna_adapter.compute_reward(action)

        # TODO: Si target dead, terminar episodio.
        if self._current_step >= self._max_episode_steps:
            self._fauna_adapter.finish_training()
            self._finished = True
            terminated = True

        return observation, reward, terminated, truncated, info

    def close(self):
        if self._fauna_adapter is not None and not self._finished:
            self._fauna_adapter.finish_training()
            self._finished = True
        super().close()

    def _get_default_observation(self):
        return {
            "vitality": np.array([0.0], dtype=np.float32),
            "age": np.array([0.0], dtype=np.float32),
            "nearby_flora": np.array([0], dtype=np.float32),
            "nearby_fauna": np.array([0], dtype=np.float32),
            "temperature": np.array([0.5], dtype=np.float32),
            "humidity": np.array([0.5], dtype=np.float32)
        }
