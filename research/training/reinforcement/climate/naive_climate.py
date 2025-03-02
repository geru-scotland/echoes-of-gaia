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
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from research.training.registry import EnvironmentRegistry
from shared.enums import Agents, BiomeType, WeatherEvent, Season
from research.training.reinforcement.climate.climate_adapter import ClimateTrainAdapter


@EnvironmentRegistry.register(Agents.Reinforcement.NAIVE_CLIMATE)
class NaiveClimateEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(len(WeatherEvent))

        self.observation_space = gym.spaces.Dict({
           "temperature": gym.spaces.Box(low=-30, high=50, shape=(1,), dtype=np.float32),
            "season": gym.spaces.Discrete(len(Season))
        })

        print(self.observation_space)

        self._current_step = 0
        self._max_episode_steps = 100

        self._climate_adapter = ClimateTrainAdapter(BiomeType.SAVANNA)


    def reset(self, *, seed=None, options=None) -> ObsType:
       super().reset(seed=seed, options=options)
       self._current_step = 0
       self._climate_adapter = ClimateTrainAdapter(BiomeType.TROPICAL)

       return self._climate_adapter.get_observation()

    def step(self, action):
        # TODO: Tensorboard
        # NORMALIZAR VALORES, IMPORTANTE
        # hay que hacer conversión de action a weatherevent, no sé qué me devuelve el modelo,
        # asumo que idx de acción
        terminated = False
        truncated = self._current_step >= self._max_episode_steps
        self._climate_adapter.progress_climate(action)