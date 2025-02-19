"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from logging import Logger
from abc import abstractmethod, ABC
from typing import Optional, Callable

from simpy import Environment as simpyEnv

from shared.enums import ComponentType
from shared.strings import Loggers
from utils.loggers import LoggerManager


class Component(ABC):
    def __init__(self, type: ComponentType, env: simpyEnv):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._type: ComponentType = type
        self._env: simpyEnv = env

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def _update(self, delay: Optional[int] = None):
        pass

    @property
    def type(self):
        return self._type


class BiomeComponent(Component):
    def __init__(self, type: ComponentType, env: simpyEnv):
        super().__init__(type, env)

    def get_state(self):
        pass

    def _update(self, delay: Optional[int] = None):
        pass


class EntityComponent(Component):
    def __init__(self, type: ComponentType, env: simpyEnv, callback: Callable):
        super().__init__(type, env)
        self._handle_component_update: Callable = callback 

    def get_state(self):
        pass

    def _update(self, delay: Optional[int] = None):
        pass





