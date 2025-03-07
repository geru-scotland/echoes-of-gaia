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
from typing import Any, Dict, Type

from simpy import Environment as simpyEnv

from biome.entities.descriptor import EntityDescriptor
from biome.entities.state import EntityState
from biome.systems.state.handler import StateHandler
from shared.enums import ComponentType
from shared.strings import Loggers
from shared.types import ComponentDict, HabitatList
from simulation.core.systems.events.handler import EventHandler
from biome.components.base.component import EntityComponent
from utils.loggers import LoggerManager


class Entity(EventHandler, StateHandler, ABC):

    def __init__(self, id: int, env: simpyEnv, descriptor: EntityDescriptor, habitats: HabitatList):
        super().__init__()
        self._id: int = id
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._descriptor: EntityDescriptor = descriptor
        self._env: simpyEnv = env
        self._components: ComponentDict = {}
        self._habitats: HabitatList = habitats
        self._state: EntityState = EntityState()

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

    def handle_component_update(self, component_class: Type, **kwargs: Any):
        if kwargs:
            self._logger.debug(f"Updating entity: {self._descriptor.species} ({self._descriptor.entity_type}),"
                               f" [component: {component_class.__name__}]: {kwargs}")
            for key, value in kwargs.items():
                self._state.update(key, value)

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._state

    def get_attribute(self, attribute: str) -> Any:
        return self._state.get(attribute)

    def get_state_fields(self) -> Dict[str, Any]:
        return self._state.fields

    @abstractmethod
    def dump_components(self) -> None:
        raise NotImplementedError

    def get_type(self):
        return self._descriptor.entity_type

    def get_species(self):
        return self._descriptor.species

    @property
    def type(self):
        return self.get_type()

    def update(self):
        pass