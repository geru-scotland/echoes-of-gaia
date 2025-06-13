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

"""
Entity spawning system with component assembly and position allocation.

Creates flora and fauna entities with custom component configurations;
handles dynamic component instantiation and entity registration.
Manages initial spawns and evolutionary entity creation - integrates
with map allocator for habitat-based positioning and entity lifecycle.
"""

import itertools
import random
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

from simpy import Environment as simpyEnv

from biome.components.environmental.weather_adaptation import WeatherAdaptationComponent
from biome.components.physiological.autotrophic_nutrition import (
    AutotrophicNutritionComponent,
)
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.heterotrophic_nutrition import (
    HeterotrophicNutritionComponent,
)
from biome.components.physiological.photosynthetic_metabolism import (
    PhotosyntheticMetabolismComponent,
)
from biome.components.physiological.vital import VitalComponent
from biome.components.registry import get_component_class
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.entities.flora import Flora
from biome.systems.maps.map_allocator import MapAllocator
from shared.enums.enums import DietType, FaunaSpecies, FloraSpecies
from shared.enums.strings import Loggers
from shared.stores.biome_store import BiomeStore
from shared.types import EntityDefinitions, EntityRegistry, HabitatList
from utils.loggers import LoggerManager
from utils.middleware import log_execution_time


class SpawnSystem:
    def __init__(self, env: simpyEnv, map_allocator: MapAllocator):
        self._flora_registry: EntityRegistry = {}
        self._fauna_registry: EntityRegistry = {}
        self._main_registry: EntityRegistry = {}
        self._id_generator = itertools.count(0)
        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._env: simpyEnv = env
        self._map_allocator: MapAllocator = map_allocator

    def _create_single_entity(self, entity_class, entity_species, habitats: HabitatList,
                              lifespan: float, components: Dict[str, Dict[str, Any]], evolution_cycle: int = 0,
                              diet_type: DietType = None, biome_store: Optional[Dict[str, Any]] = None) -> Optional[
        Entity]:
        entity_id: int = next(self._id_generator)

        if entity_class == Fauna and diet_type:
            entity: Entity = entity_class(entity_id, self._env, entity_species, habitats, lifespan,
                                          diet_type=diet_type, evolution_cycle=evolution_cycle)
        else:
            entity: Entity = entity_class(entity_id, self._env, entity_species, habitats, lifespan, evolution_cycle)

        if not components:
            return self._finalize_entity_creation(entity)

        self._add_components_to_entity(entity, components, lifespan)

        return self._finalize_entity_creation(entity)

    def _add_components_to_entity(self, entity: Entity, components_dict: Dict[str, Dict[str, Any]],
                                  lifespan: float) -> None:
        for class_name, params in components_dict.items():
            component_class = get_component_class(class_name)
            if not component_class:
                self._logger.debug(f"Class not found: {class_name}")
                continue

            data = params.copy() if params else {}

            if self._requires_lifespan(component_class):
                data["lifespan"] = lifespan

            try:
                component_instance = component_class(self._env, entity.event_notifier, **data)
                self._logger.debug(
                    f"Adding componente {component_instance.__class__.__name__} a {entity.get_type()}")
                entity.add_component(component_instance)
            except Exception as e:
                self._logger.error(f"Error creating the component {class_name}: {e}", exc_info=True)

    def _get_component_defaults(self, class_name: str, entity_species: str, biome_store: Dict[str, Any]) -> Dict:

        defaults = BiomeStore.components.get(class_name, {}).get("defaults", {}).copy()

        species_components = biome_store.get(entity_species, {}).get("components", {})
        entity_specific = species_components.get(class_name, {})

        if entity_specific:
            defaults.update(entity_specific)

        for key, value in defaults.items():
            if isinstance(value, (int, float)) and value not in (0.0, 1.0):
                variation = random.uniform(0.7, 1.3)
                defaults[key] = value * variation

        return defaults

    def _requires_lifespan(self, component_class) -> bool:
        return component_class in [
            GrowthComponent, VitalComponent, PhotosyntheticMetabolismComponent,
            WeatherAdaptationComponent, AutotrophicNutritionComponent,
            HeterotrophicNutritionComponent
        ]

    def _finalize_entity_creation(self, entity: Entity) -> Optional[Entity]:
        try:
            self._logger.debug(f"=== CREATING ENTITY ===")
            self._logger.debug(f"ID: {entity.get_id()}")
            self._logger.debug(f"Type: {entity.get_type()}")
            self._logger.debug(f"Species: {entity.get_species()}")
            self._logger.debug(f"Habitats: {entity.get_habitats()}")

            self._logger.debug("Components:")
            for comp_type, component in entity.components.items():
                self._logger.debug(f"  - {comp_type}: {component.__class__.__name__}")
                if hasattr(component, 'event_notifier'):
                    is_valid = component.event_notifier is not None
                    self._logger.debug(f"    event_notifier: {'OK' if is_valid else 'None (POTENTIAL ERROR)'}")

                for attr_name, attr_value in vars(component).items():
                    if not attr_name.startswith('_') and not callable(attr_value):
                        self._logger.debug(f"    {attr_name}: {attr_value}")

            if self._request_allocation(entity):
                return entity

            self._logger.warning("Entity allocation failed!")
            entity.clear_and_unregister()
            return None
        except Exception as e:
            self._logger.error(f"Error finalizing entity creation: {e}", exc_info=True)
            entity.clear_and_unregister()
            return None

    @log_execution_time(context="Entities created")
    def _create_entities(self, spawns: EntityDefinitions, entity_class, entity_species_enum,
                         biome_store: Dict[str, Any]) -> EntityRegistry:
        if not spawns:
            return {}

        self._logger.info(f"Creating entities... {entity_class.__name__}...")
        entity_registry: EntityRegistry = {}

        for spawn_definition in spawns:
            try:
                species_name = str(spawn_definition.get("species", "")).lower()
                amount = spawn_definition.get("spawns", 0)
                lifespan = spawn_definition.get("avg-lifespan", random.randint(1, 20))
                custom_components = spawn_definition.get("components", [])

                self._logger.info(f"Spawning {amount} {species_name}...")

                if amount < 1 or amount > 350:
                    raise ValueError(f"Invalid spawn amount: {amount}. Must be between 1 and 150.")
                if lifespan <= 0:
                    raise ValueError(f"Invalid avg-lifespan amount: {lifespan}. Must be higher than 0")

                for _ in range(amount):
                    entity = self.spawn(
                        entity_class=entity_class,
                        entity_species_enum=entity_species_enum,
                        species_name=species_name,
                        lifespan=lifespan,
                        custom_components=custom_components,
                        biome_store=biome_store,
                    )

                    if entity:
                        entity_registry[entity.get_id()] = entity

            except (AttributeError, ValueError) as e:
                self._logger.exception(f"There was an error loading {entity_class.__name__} spawns: {e}")
                continue

        return entity_registry

    def _request_allocation(self, entity: Entity) -> bool:
        self._logger.debug(
            f"Adding {entity.get_type()} to index map: {entity.get_id()}, habitats: {entity.get_habitats()}")

        position = self._map_allocator.allocate_position(entity.get_id(), entity.get_habitats())

        if position is None:
            self._logger.error(
                f"Failed to allocate position for entity {entity.get_type()} ({entity.get_species()})(id: {entity.get_id()})!")
            return False

        entity.set_position(position[0], position[1])
        return True

    def spawn(self, entity_class, entity_species_enum, species_name: str, lifespan: float = 20.0,
              custom_components: List[Dict] = None, biome_store=None, evolution_cycle: int = 0) -> Optional[Entity]:
        # 1. Biome_store apropiado
        biome_store = self._get_biome_store(entity_class, biome_store)
        if biome_store is None:
            return None

        # 2. Especie y hábitats
        try:
            species_data = self._get_species_data(entity_species_enum, species_name, biome_store)
            if not species_data:
                return None
            entity_species, habitats, diet = species_data
        except Exception as e:
            self._logger.error(f"Error obtaining species data: {e}")
            return None

        # 3. Preparo los componentes
        components = self._prepare_components(entity_species, entity_class, custom_components, biome_store)

        # 4. Instancio la entidad
        entity = self._create_single_entity(entity_class, entity_species, habitats, lifespan, components,
                                            evolution_cycle, diet, biome_store)

        # 5. Registro la entida en main registry, flora, fauna etc.
        if entity:
            self._register_entity(entity, entity_class)
            return entity

        self._logger.warning("Entity NONE after create_single_entity (SpawnSystem)")
        return None

    def _get_biome_store(self, entity_class, biome_store=None):
        if biome_store is not None:
            return biome_store

        if entity_class == Flora:
            return BiomeStore.flora
        elif entity_class == Fauna:
            return BiomeStore.fauna
        else:
            self._logger.error(f"Unknown entity class: {entity_class}")
            return None

    def _get_species_data(self, entity_species_enum, species_name: str, biome_store) -> Tuple[
        Any, HabitatList, DietType]:
        entity_species = entity_species_enum(species_name.lower())
        habitats: HabitatList = biome_store.get(entity_species, {}).get("habitat", {})

        diet_type = DietType.HERBIVORE
        if biome_store == BiomeStore.fauna:
            diet_str = biome_store.get(entity_species, {}).get("diet", "herbivore")
            try:
                diet_type = DietType(diet_str)
            except ValueError:
                self._logger.warning(f"Invalid diet type '{diet_str}', using default: herbivore")

        return entity_species, habitats, diet_type

    def _prepare_components(self, entity_species, entity_class, custom_components: List[Dict],
                            biome_store: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        components = {}

        entity_type = "flora" if entity_class == Flora else "fauna"

        fixed_components = BiomeStore.components.get("fixed_components", {}).get(entity_type, {})
        fixed_component_names = list(fixed_components.keys())

        species_components = biome_store.get(entity_species, {}).get("components", {})

        custom_component_map = {}
        if custom_components:
            for comp_dict in custom_components:
                for comp_name, comp_params in comp_dict.items():
                    custom_component_map[comp_name] = comp_params

        for comp_name in fixed_component_names:
            default_values = BiomeStore.components.get(comp_name, {}).get("defaults", {})
            if default_values is None:
                default_values = {}
            else:
                default_values = default_values.copy()

            components[comp_name] = default_values

            if comp_name in species_components:
                if species_components[comp_name] is not None:
                    components[comp_name].update(species_components[comp_name])

            if comp_name in custom_component_map:
                if custom_component_map[comp_name] is not None:
                    components[comp_name].update(custom_component_map[comp_name])

        for comp_name, comp_params in species_components.items():
            if comp_name not in components:
                default_values = BiomeStore.components.get(comp_name, {}).get("defaults", {})
                if default_values is None:
                    default_values = {}
                else:
                    default_values = default_values.copy()

                components[comp_name] = default_values

                if comp_params is not None:
                    components[comp_name].update(comp_params)

                if comp_name in custom_component_map and custom_component_map[comp_name] is not None:
                    components[comp_name].update(custom_component_map[comp_name])

        for comp_name, comp_params in custom_component_map.items():
            if comp_name not in components:
                default_values = BiomeStore.components.get(comp_name, {}).get("defaults", {})
                if default_values is None:
                    default_values = {}
                else:
                    default_values = default_values.copy()

                components[comp_name] = default_values

                if comp_params is not None:
                    components[comp_name].update(comp_params)

        return components

    def _register_entity(self, entity: Entity, entity_class):
        if entity_class == Flora:
            self._flora_registry[entity.get_id()] = entity
        elif entity_class == Fauna:
            self._fauna_registry[entity.get_id()] = entity

        self._main_registry[entity.get_id()] = entity

    def initial_spawns(self, flora_spawns: EntityDefinitions = None, fauna_spawns: EntityDefinitions = None):
        self._flora_registry = self._create_entities(flora_spawns, Flora, FloraSpecies, BiomeStore.flora)
        self._fauna_registry = self._create_entities(fauna_spawns, Fauna, FaunaSpecies, BiomeStore.fauna)
        self._main_registry = {**self._flora_registry, **self._fauna_registry}

    def get_entity_registry(self) -> EntityRegistry:
        return self._main_registry

    @property
    def flora_registry(self) -> EntityRegistry:
        return self._flora_registry

    @property
    def fauna_registry(self) -> EntityRegistry:
        return self._fauna_registry
