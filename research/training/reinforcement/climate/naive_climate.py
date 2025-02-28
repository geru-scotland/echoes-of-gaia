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

import gymnasium as gym
from gymnasium.core import ObsType

from biome.systems.climate.seasons import SeasonSystem
from research.training.registry import EnvironmentRegistry
from shared.enums import Agents


@EnvironmentRegistry.register(Agents.Reinforcement.NAIVE_CLIMATE)
class NaiveClimateEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self._season_system = SeasonSystem

    def reset(self, *, seed=None, options=None) -> ObsType:
       super().reset(seed=seed, options=options)
       observation: ObsType = self.observation_space.sample()
       return self.observation_space

    def step(self, action):
        pass