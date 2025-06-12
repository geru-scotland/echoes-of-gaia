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
Gymnasium environment for climate control reinforcement learning.

Implements a basic climate simulation environment with seasonality and biome
selection; defines observation and action spaces for weather event selection.
Episodes span a full simulated year - uses climate adapter to manage state
transitions and reward computation for maintaining optimal conditions.
"""

from logging import Logger

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from research.training.registry import EnvironmentRegistry
from shared.enums.enums import Agents, BiomeType, WeatherEvent, Season
from research.training.reinforcement.climate.climate_adapter import ClimateTrainAdapter
from shared.enums.strings import Loggers
from shared.timers import Timers
from utils.loggers import LoggerManager


@EnvironmentRegistry.register(Agents.Reinforcement.NAIVE_CLIMATE)
class NaiveClimateEnvironment(gym.Env):
    def __init__(self, local_fov_config=None):
        super().__init__()

        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self.np_random, _ = seeding.np_random(None)
        self.action_space: Discrete = gym.spaces.Discrete(len(WeatherEvent))

        self.observation_space = gym.spaces.Dict({
            "temperature": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "humidity": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "precipitation": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "biome_type": gym.spaces.Discrete(len(BiomeType)),
            "season": gym.spaces.Discrete(len(Season))
        })

        self._current_step: int = 0
        self._max_episode_steps: int = Timers.Calendar.YEAR
        self._climate_adapter: ClimateTrainAdapter = ClimateTrainAdapter(BiomeType.SAVANNA)

    def reset(self, *, seed=None, options=None) -> ObsType:
        super().reset(seed=seed, options=options)
        self._current_step = 0

        random_biome = self.np_random.choice(list(BiomeType))
        self._logger.debug(f"NEW BIOMA: {random_biome}")
        self._climate_adapter = ClimateTrainAdapter(random_biome)
        self._logger.debug(f"Initial Data: {self._climate_adapter._state}")
        return self._climate_adapter.get_observation(), {}

    def step(self, action):
        # TODO: Tensorboar5d
        # NORMALIZAR VALORES, IMPORTANTE
        # hay que hacer conversión de action a weatherevent, no sé qué me devuelve el modelo,
        # asumo que idx de acción

        self._current_step += 1
        self._climate_adapter.progress_climate(action)

        terminated = False
        truncated = self._current_step >= self._max_episode_steps

        observation: ObsType = self._climate_adapter.get_observation()
        reward: float = self._climate_adapter.compute_reward(action)

        return observation, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
