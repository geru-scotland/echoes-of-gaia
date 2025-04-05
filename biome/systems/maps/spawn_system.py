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
from typing import List, Dict, Optional

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
from shared.enums.enums import FloraSpecies, FaunaSpecies
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
                              lifespan: float, components: List[Dict], evolution_cycle: int = 0) -> Optional[Entity]:
        entity_id: int = next(self._id_generator)
        entity: Entity = entity_class(entity_id, self._env, entity_species, habitats, lifespan, evolution_cycle)

        if not components:
            if self._request_allocation(entity):
                return entity
            self._logger.warning(f"Request allocation denied for entity class {entity_class}")
            return None

        for component in components:
            # TODO: Hacer que, a parte de en el bioime.yaml, revise en ecosystem para los componentes
            # de la entidad, biomestore. importante, es TAMBIÉN.
            for class_name, attribute_dict in component.items():
                defaults = BiomeStore.components.get(class_name, {}).get("defaults", {}).copy()

                for key, value in defaults.items():
                    if isinstance(value, (int, float)):
                        # Variación del 10%
                        if value != 1.0 or value != 0.0:
                            variation = random.uniform(0.7, 1.3)
                            defaults[key] = value * variation

                # Primera vez con esta sintaxis
                # desempaqueto defaults, y luego, los atributos del value del dict
                # esos atributos, van a sobreescribir, asi que es un merge
                # lo que no esté en uno, está en el otro, y los valores finales son los de attrib.
                data = {**defaults, **(attribute_dict or {})}

                if data:
                    component_class = get_component_class(class_name)

                    if component_class in [GrowthComponent, VitalComponent, PhotosyntheticMetabolismComponent,
                                           WeatherAdaptationComponent,
                                           AutotrophicNutritionComponent, HeterotrophicNutritionComponent]:
                        data.update({"lifespan": lifespan})

                    if component_class:
                        component_instance = component_class(self._env, entity.event_notifier, **data)

                        self._logger.debug(
                            f"ADDING COMPONENT {component_instance.__class__} to {entity.type}")
                        entity.add_component(component_instance)
                    else:
                        self._logger.debug(f"Class not found: {class_name}")

        if self._request_allocation(entity):
            return entity

        # TODO: Borrar componentes creados
        self._logger.warning("Entity was not created!")
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
        if biome_store is None:
            if entity_class == Flora:
                biome_store = BiomeStore.flora
            elif entity_class == Fauna:
                biome_store = BiomeStore.fauna
            else:
                self._logger.error(f"Unknown entity class: {entity_class}")
                return None

        try:
            entity_species = entity_species_enum(species_name.lower())
            habitats: HabitatList = biome_store.get(entity_species, {}).get("habitat", {})
        except (ValueError, AttributeError) as e:
            self._logger.error(f"Invalid species: {species_name}. Error: {e}")
            return None

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

        entity = self._create_single_entity(entity_class, entity_species, habitats, lifespan, components,
                                            evolution_cycle)

        if entity:
            if entity_class == Flora:
                self._flora_registry[entity.get_id()] = entity
            elif entity_class == Fauna:
                self._fauna_registry[entity.get_id()] = entity

            self._main_registry[entity.get_id()] = entity

            return entity

        self._logger.warning("Entity NONE after create_single_entity (SpawnSystem)")
        return None

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
