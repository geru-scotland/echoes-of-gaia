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
from typing import Any, Dict, Type, Optional

from simpy import Environment as simpyEnv

from biome.components.kinematics.movement import MovementComponent
from biome.components.physiological.vital import VitalComponent
from biome.entities.descriptor import EntityDescriptor
from biome.entities.state import EntityState
from biome.services.climate_service import ClimateService
from biome.systems.climate.system import ClimateSystem
from biome.systems.components.managers.movement_manager import MovementComponentManager
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.events.event_notifier import EventNotifier
from biome.systems.state.handler import StateHandler
from shared.enums.enums import ComponentType, EntityType, Direction
from shared.enums.events import ComponentEvent, BiomeEvent
from shared.enums.strings import Loggers
from shared.types import ComponentDict, HabitatList, Position
from shared.events.handler import EventHandler
from biome.components.base.component import EntityComponent
from utils.loggers import LoggerManager


class Entity(EventHandler, StateHandler, ABC):

    def __init__(self, id: int, env: simpyEnv, descriptor: EntityDescriptor, habitats: HabitatList, lifespan: float,
                 evolution_cycle: int = 0):
        self._event_notifier: EventNotifier = EventNotifier()
        super().__init__()
        self._id: int = id
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._descriptor: EntityDescriptor = descriptor
        self._env: simpyEnv = env
        self._components: ComponentDict = {}
        self._habitats: HabitatList = habitats
        self._birth_tick: int = self._env.now
        self._lifespan: float = lifespan

        self._state: EntityState = EntityState()
        self._state.update("is_dead", False)
        self._state.update("evolution_cycle", evolution_cycle)

    def _register_events(self):
        self._event_notifier.register(ComponentEvent.UPDATE_STATE, self._handle_component_update)
        self._event_notifier.register(ComponentEvent.ENTITY_DEATH, self._handle_death)

        # BiomeEventBus ahora
        BiomeEventBus.register(BiomeEvent.WEATHER_UPDATE, self._handle_weather_update)

    def _handle_component_update(self, component_class: Type, **kwargs: Any):
        if kwargs:
            # self._logger.debug(f"[Sim tick: {self._env.now} (called in: {kwargs.get("tick")})] Updating entity: {self._descriptor.species} (id: {self._id}) ({self._descriptor.entity_type}),"
            #                    f" [component: {component_class.__name__}]: {kwargs}")
            for key, value in kwargs.items():
                self._state.update(key, value)
                if key == "biological_age":
                    self._event_notifier.notify(ComponentEvent.BIOLOGICAL_AGE_UPDATED, biological_age=value)

    def _handle_death(self, *args, **kwargs):
        cleanup_dead_entities: bool = kwargs.get("cleanup_dead_entities", False)
        self._logger.debug(f"Entity {self._id} ({self._descriptor.species}) has died")
        self._state.update("is_dead", True)
        BiomeEventBus.trigger(BiomeEvent.ENTITY_DEATH, entity_id=self._id)
        self.clear_and_unregister(cleanup_dead_entities)

        self._state.update("death_tick", self._env.now)

    def _handle_weather_update(self, *args, **kwargs):
        self._event_notifier.notify(ComponentEvent.WEATHER_UPDATE, **kwargs)

    def add_component(self, component: EntityComponent):
        self._logger.debug(f"Adding component to {self._descriptor.species}: {component.type}")
        self._components[component.type] = component
        component.set_host(self)

    def get_component(self, type: ComponentType):
        return self._components.get(type, None)

    def get_components_size(self) -> int:
        return len(self._components)

    def get_id(self) -> int:
        return self._id

    def get_habitats(self) -> HabitatList:
        return self._habitats

    def get_position(self) -> Optional[Position]:
        transform_component = self._components.get(ComponentType.TRANSFORM)
        if transform_component:
            return transform_component.get_position()
        self._logger.warning("Trying to get TRANSFORM component from an entity that doesn't have it.")
        return None

    def set_position(self, x, y):
        self._components[ComponentType.TRANSFORM].set_position(x, y)

        if ComponentType.MOVEMENT in self._components:
            self._event_notifier.notify(ComponentEvent.POSITION_UPDATED, position=(x, y))

    def has_attribute(self, attribute: str) -> bool:
        return attribute in self._state

    def get_attribute(self, attribute: str) -> Any:
        return self._state.get(attribute)

    def get_state_fields(self) -> Dict[str, Dict[str, Any]]:
        fields: Dict[str, Any] = {
            "general": {
                "evolution_cycle": self._state.get("evolution_cycle", 0),
                "is_dead": self._state.get("is_dead", False),
                "death_tick": self._state.get("death_tick", -1),
                "Lifespan": self._lifespan,
                "is_dormant": self._state.get("is_dormant", False)
            }
        }
        vital_component: ComponentType.VITAL = self._components.get(ComponentType.VITAL, None)

        if vital_component is not None:
            fields["general"].update({"stress_level": vital_component.stress_level})

        for component_type, component in self._components.items():
            component_name = str(component_type).split('.')[-1].lower()
            component_fields = component.get_state()
            if component_fields:
                fields[component_name] = component_fields
        return fields

    def move(self, direction: Direction):
        if not self.is_alive():
            return

        if self._components[ComponentType.MOVEMENT]:
            self._components[ComponentType.MOVEMENT].move(direction)
            return

        self._logger.warning(f"Entity couldn't move to direction: {direction}!")

    @abstractmethod
    def dump_components(self) -> None:
        raise NotImplementedError

    def get_type(self) -> EntityType:
        return self._descriptor.entity_type

    def get_species(self):
        return self._descriptor.species

    def is_alive(self) -> bool:
        return not self._state.get("is_dead", False)

    def clear_and_unregister(self, clear_components: bool = False) -> None:
        BiomeEventBus.unregister(BiomeEvent.WEATHER_UPDATE, self._handle_weather_update)

        for _, component in self._components.items():
            component.disable_notifier()

        self._event_notifier.clear()
        # self.event_notifier.unregister(ComponentEvent.UPDATE_STATE, self._handle_component_update)
        # self.event_notifier.unregister(ComponentEvent.ENTITY_DEATH, self._handle_death)

        if clear_components:
            self._components = {}
        self._event_notifier = None

    @property
    def type(self):
        return self.get_type()

    @property
    def event_notifier(self):
        return self._event_notifier

    def update(self):
        pass

    @property
    def components(self):
        return self._components

    @property
    def lifespan(self):
        return self._lifespan

    @property
    def vitality(self) -> float:
        if self._components and self._components[ComponentType.VITAL]:
            return self._components[ComponentType.VITAL].vitality

    @property
    def movement_component(self) -> Optional[MovementComponent]:
        if self._components[ComponentType.MOVEMENT]:
            return self._components[ComponentType.MOVEMENT]
        return None
