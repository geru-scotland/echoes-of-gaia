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
import sys
import traceback
from logging import Logger
from typing import List, Dict

from simpy import Environment as simpyEnv

from biome.entities.entity import Entity
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.maps.map_allocator import MapAllocator
from biome.systems.maps.spawn_system import SpawnSystem
from biome.systems.maps.worldmap import WorldMap
from shared.enums.enums import EntityType
from shared.enums.events import BiomeEvent
from shared.enums.strings import Loggers
from shared.types import TileMap, EntityList, EntityDefinitions, EntityRegistry, \
    TerrainMap, EntityIndexMap, Position
from utils.loggers import LoggerManager


class WorldMapManager:

    def __init__(self, env: simpyEnv, tile_map: TileMap, flora_definitions: EntityDefinitions,
                 fauna_definitions: EntityDefinitions, remove_dead_entities: bool = False):

        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._logger.info("Creating Worldmap...")
        self._env: simpyEnv = env
        self._terrain_map: TerrainMap = tile_map
        self._map_allocator = MapAllocator(tile_map)
        self._cleanup_dead_entities: bool = remove_dead_entities

        try:
            self._spawn_system = SpawnSystem(env, self._map_allocator)
            self._spawn_system.initial_spawns(flora_definitions, fauna_definitions)

            self._entity_registry: EntityRegistry = self._spawn_system.get_entity_registry()
            self._entity_index_map: EntityIndexMap = self._map_allocator.entity_index_map

            self._world_map = WorldMap(tile_map=tile_map, entity_registry=self._entity_registry, entity_index_map=self._entity_index_map)

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error("There was an exception spawning entities: %s", tb)

    def add_entity(self, entity_class, entity_species_enum, species_name: str, lifespan: float, custom_components: List[Dict] = None, evolution_cycle: int = 0):
        try:
            entity = self._spawn_system.spawn(
                entity_class=entity_class,
                entity_species_enum=entity_species_enum,
                species_name=species_name,
                lifespan=lifespan,
                custom_components=custom_components,
                evolution_cycle=evolution_cycle
            )

            BiomeEventBus.trigger(BiomeEvent.ENTITY_CREATED, entity_class=entity_class, entity_species_enum=entity_species_enum, species_name=species_name,
                                  lifespan=lifespan, custom_components=None, evolution_cycle=evolution_cycle)
            if entity:
                self._logger.info(f"Entity added correctly ID={entity.get_id()}, Position: {entity.get_position()} Species={species_name}")
                return entity
            else:
                self._logger.warning(f"Entity coudln't be created {entity_class.__name__}, species: {species_name}")
                return None

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error(f"Error when creating entity {e}\n{tb}")
            return None

    def remove_entity(self, entity_id: int) -> bool:
        if entity_id not in self._entity_registry:
            return False

        entity = self._entity_registry[entity_id]
        position = entity.get_position()
        habitats = entity.get_habitats()

        del self._entity_registry[entity_id]

        if entity.get_type() == EntityType.FLORA:
            if entity_id in self._spawn_system.flora_registry:
                del self._spawn_system.flora_registry[entity_id]
        elif entity.get_type() == EntityType.FAUNA:
            if entity_id in self._spawn_system.fauna_registry:
                del self._spawn_system.fauna_registry[entity_id]

        if position is not None:
            self._map_allocator.free_position(position, habitats)

        entity.clear_and_unregister(clear_components=True)

        return True

    def handle_entity_death(self, entity_id: int) -> None:
        if not self._cleanup_dead_entities:
            self._logger.info(f"Entity {entity_id} died, but remains in world (cleanup disabled)")
            return

        if self.remove_entity(entity_id):
            self._logger.info(f"Entity {entity_id} died and was cleaned up")

    def move_entity(self, entity_id: int, new_position: Position) -> bool:
        if entity_id not in self._entity_registry:
            return False

        entity = self._entity_registry[entity_id]
        current_position = entity.get_position()

        if self._map_allocator.move_entity(entity_id, current_position, new_position, entity.get_habitats()):
            entity.set_position(new_position[0], new_position[1])
            return True

        return False

    def get_world_map(self):
        return self._world_map

    def get_entities(self) -> EntityList:
        return self._world_map.get_entities()

    def _is_valid_position(self):
        pass
