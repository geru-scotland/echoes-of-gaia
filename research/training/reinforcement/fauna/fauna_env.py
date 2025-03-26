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

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete

from shared.enums.enums import FaunaAction
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class FaunaEnvironment(gym.Env):

    def __init__(self):
        super().__init__()
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self.action_space: Discrete = gym.spaces.Discrete(len(FaunaAction))
        self.observation_space = spaces.Dict({
        })

    def reset(self, *, seed=None, options=None) -> ObsType:
        super().reset(seed=seed, options=options)
        pass

    def step(self, action):
        pass
        # return observation, reward, terminated, truncated, {}
