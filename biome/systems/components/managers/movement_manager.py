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

from simpy import Environment as simpyEnv

from biome.components.kinematics.movement import MovementComponent
from biome.systems.components.managers.base import BaseComponentManager
from shared.enums.enums import Direction
from shared.enums.strings import Loggers
from shared.timers import Timers
from utils.loggers import LoggerManager
from collections import deque

from collections import deque


class MovementComponentManager(BaseComponentManager[MovementComponent]):
    def __init__(self, env: simpyEnv):
        super().__init__(env)
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._env.process(self.move(Timers.Calendar.DAY))
        self._counter = 0
