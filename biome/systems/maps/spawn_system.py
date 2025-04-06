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
import itertools
import random
from logging import Logger
from typing import List, Dict, Optional, Tuple, Any

from simpy import Environment as simpyEnv

from biome.components.environmental.weather_adaptation import WeatherAdaptationComponent
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.heterotrophic_nutrition import HeterotrophicNutritionComponent
from biome.components.physiological.photosynthetic_metabolism import PhotosyntheticMetabolismComponent
from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from biome.components.physiological.vital import VitalComponent
from biome.components.registry import get_component_class
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.entities.flora import Flora
from biome.systems.maps.map_allocator import MapAllocator
from shared.enums.enums import FloraSpecies, FaunaSpecies, DietType
from shared.enums.strings import Loggers
from shared.stores.biome_store import BiomeStore
from shared.types import EntityRegistry, EntityDefinitions, HabitatList
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
                              lifespan: float, components: List[Dict], evolution_cycle: int = 0,
                              diet_type: DietType = None) -> Optional[Entity]:
        entity_id: int = next(self._id_generator)

        if entity_class == Fauna and diet_type:
            entity: Entity = entity_class(entity_id, self._env, entity_species, habitats, lifespan,
                                          diet_type=diet_type, evolution_cycle=evolution_cycle)
        else:
            entity: Entity = entity_class(entity_id, self._env, entity_species, habitats, lifespan, evolution_cycle)

        if not components:
            return self._finalize_entity_creation(entity)

        for component in components:
            self._add_components_to_entity(entity, component, lifespan)

        return self._finalize_entity_creation(entity)

    def _add_components_to_entity(self, entity: Entity, component_dict: Dict, lifespan: float) -> None:
        for class_name, attribute_dict in component_dict.items():
            defaults = self._get_component_defaults(class_name)
            data = {**defaults, **(attribute_dict or {})}

            if not data:
                continue

            component_class = get_component_class(class_name)
            if not component_class:
                self._logger.debug(f"Class not found: {class_name}")
                continue

            if self._requires_lifespan(component_class):
                data.update({"lifespan": lifespan})

            try:
                component_instance = component_class(self._env, entity.event_notifier, **data)
                self._logger.debug(f"ADDING COMPONENT {component_instance.__class__} to {entity.type}")
                entity.add_component(component_instance)
            except Exception as e:
                self._logger.error(f"Error creating component {class_name}: {e}")

    def _get_component_defaults(self, class_name: str) -> Dict:
        defaults = BiomeStore.components.get(class_name, {}).get("defaults", {}).copy()

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
        if self._request_allocation(entity):
            return entity

        self._logger.warning("Entity allocation failed!")
        entity.clear_and_unregister()
        return None

    @log_execution_time(context="Entities created")
    def _create_entities(self, spawns: EntityDefinitions, entity_class, entity_species_enum,
                         biome_store) -> EntityRegistry:
        if not spawns:
            return {}

        self._logger.info(f"Creating entities... {entity_class.__name__}...")
        entity_registry: EntityRegistry = {}

        for spawn in spawns:
            try:
                entity_species = entity_species_enum(str(spawn.get("species")).lower())
                habitats: HabitatList = biome_store.get(entity_species, {}).get("habitat", {})
                amount: int = spawn.get("spawns")
                lifespan: float = spawn.get("avg-lifespan", random.randint(1, 20))

                diet_type = DietType.HERBIVORE
                if entity_class == Fauna and "diet" in spawn:
                    diet_str = spawn.get("diet", "herbivore").lower()
                    try:
                        diet_type = DietType(diet_str)
                        self._logger.debug(f"Diet type set to {diet_type} for {entity_species}")
                    except ValueError:
                        self._logger.warning(f"Invalid diet type '{diet_str}', using default: herbivore")

                if amount < 1 or amount > 350:
                    raise ValueError(f"Invalid spawn amount: {amount}. Must be between 1 and 150.")
                if lifespan <= 0:
                    raise ValueError(f"Invalid avg-lifespan amount: {lifespan}. Must be higher than 0")
            except (AttributeError, ValueError) as e:
                self._logger.exception(f"There was an error loading {entity_class.__name__} spawns: {e}")
                continue

            components = spawn.get("components", [])
            fixed_components = BiomeStore.components.get("fixed_components", [])
            components.append(fixed_components)

            if not components:
                store_components: List[str] = biome_store.get(entity_species, {}).get("components", [])
                if store_components:
                    components = [
                        {cmp: BiomeStore.components.get(cmp, {}).get("defaults", {})}
                        for cmp in store_components
                        if BiomeStore.components.get(cmp)
                    ]

            for _ in range(amount):
                if entity_class == Fauna:
                    entity = self._create_single_entity(
                        entity_class, entity_species, habitats, lifespan, components, diet_type=diet_type)
                else:
                    entity = self._create_single_entity(
                        entity_class, entity_species, habitats, lifespan, components)

                if entity:
                    entity_registry[entity.get_id()] = entity

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
        components = self._prepare_components(entity_species, custom_components, biome_store)

        # 4. Instancio la entidad
        entity = self._create_single_entity(entity_class, entity_species, habitats, lifespan, components,
                                            evolution_cycle, diet)

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

    def _prepare_components(self, entity_species, custom_components: List[Dict], biome_store) -> List[Dict]:
        components = []

        if custom_components:
            components.extend(custom_components)

        fixed_components = BiomeStore.components.get("fixed_components", [])
        components.append(fixed_components)

        if not components or (len(components) == 1 and components[0] == fixed_components):
            store_components: List[str] = biome_store.get(entity_species, {}).get("components", [])
            if store_components:
                components = [
                    {cmp: BiomeStore.components.get(cmp, {}).get("defaults", {})}
                    for cmp in store_components
                    if BiomeStore.components.get(cmp)
                ]
                components.append(fixed_components)

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
