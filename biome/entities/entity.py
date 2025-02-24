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
from typing import Any, Optional, Dict

from simpy import Environment as simpyEnv

from biome.entities.state import EntityState
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
        self._env: simpyEnv = env
        self._components: ComponentDict = {}
        self._habitats: HabitatList = habitats
        self.state: EntityState = EntityState()

    def _register_events(self):
        pass

    def add_component(self, component: EntityComponent):
        self._components[component.type] = component
        component.set_update_callback(self.handle_component_update)

    def get_component(self, type: ComponentType):
        return self._components.get(type, None)

    def get_components_size(self) -> int:
        return len(self._components)

    def get_id(self) -> int:
        return self._id

    def get_habitats(self) -> HabitatList:
        return self._habitats

    def get_position(self):
        return self._components[ComponentType.TRANSFORM].get_position()

    def set_position(self, x, y):
        self._components[ComponentType.TRANSFORM].set_position(x, y)

    def handle_component_update(self, **kwargs: Any):
        for key, value in kwargs.items():
            self.state.update(key, value)

    @abstractmethod
    def dump_components(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_type(self):
        raise NotImplementedError

    def update(self):
        pass