from typing import Dict

from shared.enums import ComponentType, EntityType
from simulation.core.systems.events.handler import EventHandler
from biome.components.component import Component

class Entity(EventHandler):

    def __init__(self, type: EntityType):
        super().__init__()
        self._type = type
        self._components: Dict[ComponentType, Component] = {}

    def _register_events(self):
        pass

    def add_component(self, component: Component):
        self._components[component.component_type] = component
        component.entity = self

    def get_component(self, type: ComponentType):
        return self._components.get(type, None)

    def update(self):
        pass