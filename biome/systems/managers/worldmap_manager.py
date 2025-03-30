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
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
from simpy import Environment as simpyEnv

from biome.entities.entity import Entity
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.maps.map_allocator import MapAllocator
from biome.systems.maps.spawn_system import SpawnSystem
from biome.systems.maps.worldmap import WorldMap
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import EntityType, FaunaSpecies, TerrainType, PositionNotValidReason
from shared.enums.events import BiomeEvent, SimulationEvent
from shared.enums.strings import Loggers
from shared.types import TileMap, EntityList, EntityDefinitions, EntityRegistry, \
    TerrainMap, EntityIndexMap, Position
from simulation.core.systems.events.event_bus import SimulationEventBus
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

            self._world_map = WorldMap(tile_map=tile_map, entity_registry=self._entity_registry,
                                       entity_index_map=self._entity_index_map)

            BiomeEventBus.register(BiomeEvent.MOVE_ENTITY, self._handle_move_entity)
            BiomeEventBus.register(BiomeEvent.VALIDATE_MOVEMENT, self._handle_validate_movement)

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error("There was an exception spawning entities: %s", tb)

    def _handle_move_entity(self, entity_id: int, new_position: Position, success_callback=None):
        success = self.move_entity(entity_id, new_position)

        if success_callback:
            success_callback(success)

        return success

    def _is_traversable_position(self, position: Position) -> bool:
        if self._map_allocator.is_position_valid(position):
            terrain_type = self._world_map.terrain_map[position]
            non_traversable_terrains = {
                int(TerrainType.WATER_DEEP),
                int(TerrainType.WATER_MID),
            }

            if int(terrain_type) in non_traversable_terrains:
                self._logger.warning(f"Terrain at position {position} is not traversable")
                return False

            return True
        return False

    def _is_valid_position(self, position: Position, entity_id: Optional[int] = None) -> Tuple[
        bool, PositionNotValidReason]:

        if not self._map_allocator.is_position_valid(position):
            self._logger.warning(f"Target position {position} is outside map boundaries")
            return False, PositionNotValidReason.POSITION_OUT_OF_BOUNDARIES

        if entity_id and self._map_allocator.get_entity_at(position) != -1:
            occupier_id = self._map_allocator.get_entity_at(position)
            if occupier_id != entity_id:
                self._logger.warning(f"Target position {position} is already occupied by entity {occupier_id}")
                return False, PositionNotValidReason.POSITION_BUSY

        if not self._is_traversable_position(position):
            return False, PositionNotValidReason.POSITION_NON_TRAVERSABLE

        return True, PositionNotValidReason.NONE

    def _handle_validate_movement(self, entity_id: int, new_position: Position, result_callback: Callable) -> None:
        if entity_id not in self._entity_registry:
            self._logger.warning(f"Entity {entity_id} not found in registry")
            result_callback(False)
            return

        result_callback(self._is_valid_position(new_position, entity_id))

    def add_entity(self, entity_class, entity_species_enum, species_name: str, lifespan: float,
                   custom_components: List[Dict] = None, evolution_cycle: int = 0):
        try:
            entity = self._spawn_system.spawn(
                entity_class=entity_class,
                entity_species_enum=entity_species_enum,
                species_name=species_name,
                lifespan=lifespan,
                custom_components=custom_components,
                evolution_cycle=evolution_cycle
            )

            self._logger.info(f"Creating evolved entity: {entity_class} with ref: {id(entity)}")

            BiomeEventBus.trigger(BiomeEvent.ENTITY_CREATED, species_name=species_name, evolution_cycle=evolution_cycle)

            if entity and TrainingTargetManager.is_training_mode() and not TrainingTargetManager.is_acquired() and entity_species_enum == FaunaSpecies:
                self._logger.info(f"{TrainingTargetManager.get_target()}")
                if TrainingTargetManager.is_valid_target(entity.get_species(), entity.get_type(), evolution_cycle):
                    self._logger.info(f"Selected entity with REF: with ref: {id(entity)}")
                    SimulationEventBus.trigger(SimulationEvent.SIMULATION_TRAIN_TARGET_ACQUIRED, entity=entity,
                                               generation=evolution_cycle)

            if entity:
                self._logger.debug(
                    f"Entity added correctly ID={entity.get_id()}, Position: {entity.get_position()} Species={species_name}")
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

        return True

    def handle_entity_death(self, entity_id: int) -> None:
        if not self._cleanup_dead_entities:
            self._logger.debug(f"Entity {entity_id} died, but remains in world (cleanup disabled)")
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

    def is_valid_position(self, position: Position, entity_id: Optional[int] = None) -> bool:
        return self._is_valid_position(position, entity_id)
