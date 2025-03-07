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
from abc import abstractmethod
from logging import Logger
from typing import TypeVar, Generic

from simpy import Environment as simpyEnv

from biome.components.base.component import Component
from shared.enums import ComponentType
from shared.types import EntityList, ComponentDict
from simulation.core.bootstrap.context.context_data import ContextData
from utils.loggers import LoggerManager

T = TypeVar("T", bound=ContextData)


class Environment(Generic[T]):

    def __init__(self, context: T, env: simpyEnv):
        self._context: T = context
        self._env: simpyEnv = env
        self._logger: Logger = LoggerManager.get_logger(self._context.logger_name)
        self._entities: EntityList = []
        self._components: ComponentDict = {}

        try:
            self._initialize_environment()
            self._logger.info(f"{self.__class__.__name__} initialized successfully!")
        except Exception as e:
            self._logger.error(f"Error initializing {self.__class__.__name__}: {e}")

    def _initialize_environment(self):
        self._logger.info(f"Setting up environment: {self.__class__.__name__}")

    @abstractmethod
    def update(self, delay: int):
        pass

    def add_component(self, component: Component) -> None:
        self._components[component.type] = component
        self._logger.info(f"Component {component.type} added to {self.__class__.__name__}")

    def get_component(self, type: ComponentType) -> Component:
        return self._components.get(type, None)
