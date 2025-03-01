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
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType

from research.training.registry import EnvironmentRegistry
from shared.enums import Agents, BiomeType
from research.training.reinforcement.climate.climate_adapter import ClimateTrainAdapter


@EnvironmentRegistry.register(Agents.Reinforcement.NAIVE_CLIMATE)
class NaiveClimateEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        # TODO: BiomeType random en cada episocio
        self._climate = ClimateTrainAdapter(BiomeType.SAVANNA)

    def reset(self, *, seed=None, options=None) -> ObsType:
       super().reset(seed=seed, options=options)
       self._climate = ClimateTrainAdapter(BiomeType.TROPICAL)
       observation: List[float] = self._climate.get_observation()
       return np.array(observation, np.float32)

    def step(self, action):
        # TODO: Tensorboard
        # NORMALIZAR VALORES, IMPORTANTE
        pass