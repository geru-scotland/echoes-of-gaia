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
from typing import List

import numpy as np
import itertools

from numpy import ndarray
from simpy import Environment as simpyEnv
from scipy.ndimage import convolve

from biome.components.registry import get_component_class
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.entities.flora import Flora
from biome.systems.maps.worldmap import WorldMap
from shared.enums import FloraType, FaunaType, Habitats, TerrainType
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from shared.types import TileMap, EntityList, HabitatCache, BiomeStoreData, EntityDefinitions, EntityRegistry, \
    TerrainMap, EntityIndexMap, HabitatList
from utils.loggers import LoggerManager
from utils.middleware import log_execution_time


class WorldMapManager:
    class SpawnSystem:
        def __init__(self, env: simpyEnv, tile_map: TileMap ):
            self._flora_registry: EntityRegistry = {}
            self._fauna_registry: EntityRegistry = {}
            self._entity_index_map: EntityIndexMap = np.full(tile_map.shape, -1, dtype=int)
            self._id_generator = itertools.count(0)
            self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
            self._env: simpyEnv = env
            self._tile_map: TileMap = tile_map
            self._habitat_cache: HabitatCache = self._precompute_habitat_cache(BiomeStore.habitats)

        @log_execution_time(context="Entities created")
        def _create_entities(self, spawns: EntityDefinitions, entity_class, entity_type_enum,
                             biome_store) -> EntityRegistry:
            if not spawns:
                return {}

            self._logger.info(f"Creating entities... {entity_class.__name__}...")

            entity_registry: EntityRegistry = {}

            for spawn in spawns:
                try:
                    entity_type = entity_type_enum(str(spawn.get("type")).lower())
                    habitats: HabitatList = biome_store.get(entity_type, {}).get("habitat", {})
                    amount = spawn.get("spawns")

                    if amount < 1 or amount > 150:
                        raise ValueError(f"Invalid spawn amount: {amount}. Must be between 1 and 150.")

                except (AttributeError, ValueError) as e:
                    self._logger.exception(f"There was an error loading {entity_class.__name__} spawns: {e}")
                    continue

                components = spawn.get("components", [])
                fixed_components = BiomeStore.components.get("fixed_components", [])
                components.append(fixed_components)
                if not components:
                    store_components: List[str] = biome_store.get(spawn.get("type"), None).get("components", [])
                    if not store_components:
                        self._logger.warning(f"Spawn doesn't have components: {spawn}")
                        entity_registry.update({
                            (id := next(self._id_generator)): entity_class(id, self._env, entity_type, habitats)
                            for _ in range(amount)
                        })
                        continue
                    components = [
                        {cmp: BiomeStore.components.get(cmp, {}).get("defaults", {})}
                        for cmp in store_components
                        if BiomeStore.components.get(cmp)
                    ]

                for _ in range(amount):
                    id: int = next(self._id_generator)
                    entity: Entity = entity_class(id, self._env, entity_type, habitats)
                    # aqui, crear el mapeo de { id: ref }
                    # y luego, al adjudicar posiciones por habitat, en un mapa np
                    # le indico los ids, que es lo que pongo en cada celda.
                    for component in components:
                        # TODO: Hacer que, a parte de en el bioime.yaml, revise en ecosystem para los componentes
                        # de la entidad, biomestore. importante, es TAMBIÉN.
                        for class_name, attribute_dict in component.items():
                            defaults = BiomeStore.components.get(class_name, {}).get("defaults", {}).copy()
                            # Primera vez con esta sintaxis
                            # desempaqueto defaults, y luego, los atributos del value del dict
                            # esos atributos, van a sobreescribir, asi que es un merge
                            # lo que no esté en uno, está en el otro, y los valores finales son los de attrib.
                            data = {**defaults, **(attribute_dict or {})}

                            if data:
                                component_class = get_component_class(class_name)
                                if component_class:
                                    component_instance = component_class(self._env, **data)
                                    self._logger.debug(
                                        f"ADDING COMPONENT {component_instance.__class__} to {entity.type}")
                                    entity.add_component(component_instance)
                                else:
                                    self._logger.debug(f"Class not found: {class_name}")
                    entity_registry[id] = entity
                    self._add_to_index_map(entity)

            return entity_registry

        def _add_to_index_map(self, entity: Entity):
            self._logger.debug(
                f"Adding {entity.get_type()} to index map: {entity.get_id()} , habitats: {entity.get_habitats()}")

            selected_position = None
            # Buscar una posición válida en cualquier hábitat
            for habitat in entity.get_habitats():
                habitat_positions: ndarray = self._habitat_cache.get(habitat, np.array([]))

                if habitat_positions.shape[0] == 0:
                    self._logger.warning(f"Habitat: {habitat} doesn't have available tiles! Can't spawn here.")
                    continue

                if selected_position is None:
                    random_index = np.random.randint(0, habitat_positions.shape[0])
                    selected_position = habitat_positions[random_index]
                    self._logger.debug(f"Random position selected: {selected_position}")

            if selected_position is None:
                self._logger.warning(f"No valid position found for entity {entity.get_type()} ({entity.get_specific_type()})(id: {entity.get_id()})!")
                return

            # Quizá, para evitar este bucle, pasar a mapa de habitats directamente
            # que a cada celda se le asigne un habitat y listo, el resto de habitats
            # que no la tengan disponible al estar rellena
            for habitat, _ in self._habitat_cache.items():
                habitat_positions: ndarray = self._habitat_cache.get(habitat, np.array([]))
                self._habitat_cache[habitat] = np.array(
                    [pos for pos in habitat_positions if not np.array_equal(pos, selected_position)])

            self._entity_index_map[selected_position[0], selected_position[1]] = entity.get_id()
            entity.set_position(selected_position[0], selected_position[1])

        def _precompute_habitat_cache(self, habitat_data: BiomeStoreData) -> HabitatCache:
            self._logger.info("Precomputing habitat cache...")
            try:
                habitat_positions: HabitatCache = {
                    Habitats.Type(habitat): np.array([[0, 0]], dtype=np.int8) for habitat, _ in habitat_data.items()
                }
                # es np, pero con terraintype, para optimizar convierto a int8
                terrain_map: ndarray = self._tile_map.astype(np.int8)

                # para convolución una celda, cuenta sus vecinas y no así misma
                kernel_neighbour: ndarray = np.array([[1, 1, 1],
                                                      [1, 0, 1],
                                                      [1, 1, 1]])
                for habitat, rules in habitat_data.items():
                    in_terrains: ndarray = np.array([int(getattr(TerrainType, terrain))
                                                     for terrain in rules.get(Habitats.Relations.IN, {})],
                                                    dtype=np.int8)
                    nearby_terrains: ndarray = np.array([int(getattr(TerrainType, terrain))
                                                         for terrain in rules.get(Habitats.Relations.NEARBY, {})],
                                                        dtype=np.int8)
                    in_mask: ndarray = np.isin(terrain_map, in_terrains).astype(np.int8)
                    nearby_mask: ndarray = np.isin(terrain_map, nearby_terrains).astype(np.int8)
                    valid_positions: ndarray = np.empty((0, 2), dtype=np.int8)
                    if nearby_mask.any():
                        nearby_convolved: ndarray = convolve(nearby_mask.astype(np.int8), kernel_neighbour,
                                                             mode="constant", cval=0)
                        # Mecaguen la leche, qué dolor de cabeza me ha dado esto.
                        # Si una celda tiene 2 nearby, 1 & 2 dará 0. Necesito
                        # que la máscara proporcione valor true, 1, para todo aquello
                        # que sea mayor que 1... Lo hago con broadcasting y convierto a 0/1 int8
                        nearby_presence = (nearby_convolved > 0).astype(np.int8)
                        habitat_mask: ndarray = in_mask & nearby_presence
                        valid_positions = np.argwhere(habitat_mask)
                        self._logger.debug(f"Habitat {habitat}. \n {terrain_map}")
                        self._logger.debug(f"Habitat mask: \n {habitat_mask}")
                        self._logger.debug(f"Convolved map: \n {nearby_convolved}")
                        self._logger.debug(f"IN mask: \n {in_mask}")
                        self._logger.debug(f"Valid positions for  {habitat}: \n {valid_positions}")
                    elif in_mask.any():
                        valid_positions = np.argwhere(in_mask)

                    habitat_positions[Habitats.Type(habitat)] = np.array(valid_positions)

                return habitat_positions

            except Exception as e:
                self._logger.error(f"Error precomputing habitat cache: {e}")

        def _compute_index_map(self):
            pass

        def position_flora_in_world(self):
            # basarme en reglas para cada tipo de flora.
            pass

        def position_fauna_in_world(self):
            # simplemente posicines válidas
            # todos terrestres por ahora
            # quizá en un futuro hacer acuáticos.
            pass

        def spawn(self,flora_spawns: EntityDefinitions = None, fauna_spawns: EntityDefinitions = None):
            self._flora_registry = self._create_entities(flora_spawns, Flora, FloraType, BiomeStore.flora)
            self._fauna_registry = self._create_entities(fauna_spawns, Fauna, FaunaType, BiomeStore.fauna)

        def get_entity_registry(self) -> EntityRegistry:
            return {**self._flora_registry, **self._fauna_registry}

        def get_entity_index_map(self) -> EntityIndexMap:
            return self._entity_index_map


    def __init__(self, env: simpyEnv, tile_map: TileMap, flora_definitions: EntityDefinitions,
                 fauna_definitions: EntityDefinitions):
        # Quizá worldmapmanager
        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._logger.info("Creating Worldmap...")
        self._env: simpyEnv = env
        self._terrain_map: TerrainMap = tile_map
        try:
            self._spawn_system = WorldMapManager.SpawnSystem(env, tile_map)
            self._spawn_system.spawn(flora_definitions, fauna_definitions)
            # quitar habitatds de worldmap, no compete al mapa tener las especificaciones
            # quizá, como mucho, habitat cache
            self._entity_registry: EntityRegistry = self._spawn_system.get_entity_registry()
            self._entity_index_map: EntityIndexMap = self._spawn_system.get_entity_index_map()
            self._world_map = WorldMap(tile_map=tile_map, entity_registry=self._entity_registry, entity_index_map=self._entity_index_map)
        except Exception as e:
            self._logger.exception(f"There was an en exception spawning entities: {e}")

    def get_world_map(self):
        return self._world_map

    def get_entities(self) -> EntityList:
        return self._world_map.get_entities()

    def _is_valid_position(self):
        pass
