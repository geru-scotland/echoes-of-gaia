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
from abc import ABC, abstractmethod
from typing import Any, Optional

from simpy import Environment as simpyEnv

from biome.systems.state.handler import StateHandler
from shared.enums import ComponentType, EntityType, Habitats
from shared.strings import Loggers
from shared.types import ComponentDict, HabitatList
from simulation.core.systems.events.handler import EventHandler
from biome.components.component import EntityComponent
from utils.loggers import LoggerManager


class Entity(EventHandler, StateHandler, ABC):

    def __init__(self, id: int, type: EntityType, env: simpyEnv, habitats: HabitatList):
        super().__init__()
        self._id: int = id
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._entity_type: EntityType = type
        self_env: simpyEnv = env
        self._components: ComponentDict = {}
        self._habitats: HabitatList = habitats

    def _register_events(self):
        pass

    def add_component(self, component: EntityComponent):
        self._components[component.type] = component
        component.entity = self

    def get_component(self, type: ComponentType):
        return self._components.get(type, None)

    def get_components_size(self) -> int:
        return len(self._components)

    def get_id(self) -> int:
        return self._id

    def get_habitats(self) -> HabitatList:
        return self._habitats

    @abstractmethod
    def handle_component_update(self, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def dump_components(self) -> None:
        raise NotImplementedError

    def update(self):
        pass