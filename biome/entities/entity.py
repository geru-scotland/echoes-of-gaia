import logging
from abc import ABC, abstractmethod

from simpy import Environment as simpyEnv

from biome.systems.state.handler import StateHandler
from shared.enums import ComponentType, EntityType
from shared.strings import Loggers
from shared.types import ComponentDict
from simulation.core.systems.events.handler import EventHandler
from biome.components.component import Component

class Entity(EventHandler, StateHandler, ABC):

    def __init__(self, type: EntityType, env: simpyEnv):
        super().__init__()
        self._logger: logging.Logger = logging.getLogger(Loggers.BIOME)
        self._entity_type: EntityType = type
        self_env: simpyEnv = env
        self._components: ComponentDict = {}

    def _register_events(self):
        pass

    def add_component(self, component: Component):
        self._components[component.type] = component
        component.entity = self

    def get_component(self, type: ComponentType):
        return self._components.get(type, None)

    def get_components_size(self) -> int:
        return len(self._components)

    @abstractmethod
    def dump_components(self) -> None:
        raise NotImplementedError

    def update(self):
        pass