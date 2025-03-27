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

from research.training.reinforcement.adapter import EnvironmentAdapter
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class FaunaAdapter(EnvironmentAdapter):

    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)

    def get_observation(self):
        pass

    def compute_reward(self, action):
        pass
