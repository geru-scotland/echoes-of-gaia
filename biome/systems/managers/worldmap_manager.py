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
import random
import sys
import traceback
from logging import Logger
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
from simpy import Environment as simpyEnv

from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.maps.map_allocator import MapAllocator
from biome.systems.maps.spawn_system import SpawnSystem
from biome.systems.maps.worldmap import WorldMap
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import EntityType, FaunaSpecies, TerrainType, PositionNotValidReason, DietType, ComponentType, \
    FloraSpecies
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
                self._logger.debug(f"Terrain at position {position} is not traversable")
                return False

            return True
        return False

    def _is_valid_position(self, position: Position, entity_id: Optional[int] = None) -> Tuple[
        bool, PositionNotValidReason]:

        if not self._map_allocator.is_position_valid(position):
            self._logger.debug(f"Target position {position} is outside map boundaries")
            return False, PositionNotValidReason.POSITION_OUT_OF_BOUNDARIES

        if entity_id and self._map_allocator.get_entity_at(position) != -1:
            occupier_id = self._map_allocator.get_entity_at(position)
            if occupier_id != entity_id:
                self._logger.debug(f"Target position {position} is already occupied by entity {occupier_id}")
                return False, PositionNotValidReason.POSITION_BUSY

        if not self._is_traversable_position(position):
            return False, PositionNotValidReason.POSITION_NON_TRAVERSABLE

        return True, PositionNotValidReason.NONE

    def _handle_validate_movement(self, entity_id: int, new_position: Position, result_callback: Callable) -> None:
        if entity_id not in self._entity_registry:
            self._logger.warning(f"Entity {entity_id} not found in registry")
            result_callback(False, PositionNotValidReason.NONE)
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

            self._logger.debug(
                f"Creating evolved entity: {entity_class}, species {species_name} and with ref: {id(entity)}")

            BiomeEventBus.trigger(BiomeEvent.ENTITY_CREATED, species_name=species_name, evolution_cycle=evolution_cycle)

            if entity and TrainingTargetManager.is_training_mode() and not TrainingTargetManager.is_acquired() and entity_species_enum == FaunaSpecies:
                self._logger.debug(f"{TrainingTargetManager.get_target()}")
                if TrainingTargetManager.is_valid_target(entity.get_species(), entity.get_type(), evolution_cycle):
                    self._logger.debug(f"Selected entity with REF: with ref: {id(entity)}")
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

    def has_alive_entities(self, type: EntityType, species: Optional[FaunaSpecies | FloraSpecies] = None) -> bool:
        return any(
            entity.is_alive() and (species is None or entity.get_species() == species)
            for entity in self.get_entities(type)
        )

    def handle_entity_death(self, entity_id: int) -> None:
        if not self._cleanup_dead_entities:
            self._logger.debug(f"Entity {entity_id} died, but remains in world (cleanup disabled)")
            return

        if self.remove_entity(entity_id):
            self._logger.debug(f"Entity {entity_id} died and was cleaned up")

    def move_entity(self, entity_id: int, new_position: Position) -> bool:
        if entity_id not in self._entity_registry:
            return False

        entity = self._entity_registry[entity_id]
        current_position = entity.get_position()

        if self._map_allocator.move_entity(entity_id, current_position, new_position, entity.get_habitats()):
            entity.set_position(new_position[0], new_position[1])
            return True

        return False

    def get_entities_near(self, position: Position, radius: int = 1) -> Dict[EntityType, List[Entity]]:
        result = {
            EntityType.FLORA: [],
            EntityType.FAUNA: []
        }

        if not position:
            return result

        y, x = position
        map_height, map_width = self._terrain_map.shape

        y_start = max(0, y - radius)
        y_end = min(map_height, y + radius + 1)
        x_start = max(0, x - radius)
        x_end = min(map_width, x + radius + 1)

        region = self._entity_index_map[y_start:y_end, x_start:x_end]

        entity_ids = np.unique(region)
        entity_ids = entity_ids[entity_ids != -1]

        for entity_id in entity_ids:
            if entity_id == self._entity_index_map[y, x]:
                continue

            if entity_id in self._entity_registry:
                entity = self._entity_registry[entity_id]
                if entity.is_alive():
                    result[entity.get_type()].append(entity)

        return result

    def get_local_maps(self, position: Position, diet_type: DietType, species: FaunaSpecies, width: int, height: int) -> \
            Optional[
                Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        y, x = position
        map_height, map_width = self._terrain_map.shape

        if not (0 <= y < map_height and 0 <= x < map_width):
            return None

        half_width = width // 2
        half_height = height // 2

        fov_y_start = max(0, y - half_height)
        fov_y_end = min(map_height, y + half_height + 1)
        fov_x_start = max(0, x - half_width)
        fov_x_end = min(map_width, x + half_width + 1)

        local_y_start = fov_y_start - (y - half_height)
        local_y_end = local_y_start + (fov_y_end - fov_y_start)
        local_x_start = fov_x_start - (x - half_width)
        local_x_end = local_x_start + (fov_x_end - fov_x_start)

        local_terrain_map = np.full((height, width), TerrainType.UNKNWON, dtype=self._terrain_map.dtype)
        local_entity_map = np.full((height, width), -1, dtype=self._entity_index_map.dtype)

        local_terrain_map[local_y_start:local_y_end, local_x_start:local_x_end] = self._terrain_map[
                                                                                  fov_y_start:fov_y_end,
                                                                                  fov_x_start:fov_x_end
                                                                                  ]

        local_entity_map[local_y_start:local_y_end, local_x_start:local_x_end] = self._entity_index_map[
                                                                                 fov_y_start:fov_y_end,
                                                                                 fov_x_start:fov_x_end
                                                                                 ]

        validity_mask = (local_terrain_map != TerrainType.UNKNWON)
        non_traversable_indices = np.isin(local_terrain_map, [TerrainType.WATER_DEEP, TerrainType.WATER_MID])
        traversability_mask = ~non_traversable_indices
        combined_validity_mask = validity_mask & traversability_mask
        obstacle_mask = (local_entity_map != -1)
        final_traversability_mask = combined_validity_mask & (~obstacle_mask)

        flora_map = np.zeros((height, width), dtype=np.int8)
        prey_map = np.zeros((height, width), dtype=np.int8)
        predator_map = np.zeros((height, width), dtype=np.int8)
        water_map = np.zeros((height, width), dtype=np.int8)
        food_map = np.zeros((height, width), dtype=np.int8)

        water_positions = np.where(local_terrain_map == TerrainType.WATER_SHALLOW)
        if len(water_positions[0]) > 0:
            water_map[water_positions] = 1

        entity_ids = np.unique(local_entity_map)
        entity_ids = entity_ids[entity_ids != -1]

        for entity_id in entity_ids:
            entity = self._entity_registry.get(entity_id)

            if not entity or not entity.is_alive():
                continue

            entity_position = np.where(local_entity_map == entity_id)

            y_pos, x_pos = entity_position[0][0], entity_position[1][0]

            if entity.get_type() == EntityType.FLORA:
                flora_map[y_pos, x_pos] = 1
                if diet_type in [DietType.HERBIVORE, DietType.OMNIVORE]:
                    food_map[y_pos, x_pos] = 1

            elif entity.get_type() == EntityType.FAUNA:
                if entity.get_species() != species:
                    if diet_type == DietType.HERBIVORE:
                        if entity.diet_type in (DietType.CARNIVORE, DietType.OMNIVORE):
                            predator_map[y_pos, x_pos] = 1

                    elif diet_type == DietType.CARNIVORE:
                        if entity.diet_type == DietType.HERBIVORE:
                            prey_map[y_pos, x_pos] = 1
                            food_map[y_pos, x_pos] = 1
                        elif entity.diet_type in (DietType.CARNIVORE, DietType.OMNIVORE):
                            predator_map[y_pos, x_pos] = 1

                    elif diet_type == DietType.OMNIVORE:
                        if entity.diet_type == DietType.HERBIVORE:
                            prey_map[y_pos, x_pos] = 1
                            food_map[y_pos, x_pos] = 1
                        elif entity.diet_type == DietType.CARNIVORE:
                            predator_map[y_pos, x_pos] = 1
        return local_terrain_map, final_traversability_mask, flora_map, prey_map, predator_map, water_map, food_map

    def get_local_visited_map(self, entity: Fauna, fov_width: int, fov_height: int) -> np.ndarray:
        position: Position = entity.get_position()
        center_y, center_x = fov_width // 2, fov_height // 2

        visited_mask = np.zeros((fov_width, fov_height), dtype=np.bool_)

        for y in range(fov_width):
            for x in range(fov_height):
                global_y = position[0] + (y - center_y)
                global_x = position[1] + (x - center_x)
                global_pos = (global_y, global_x)

                if global_pos in entity.visited_positions:
                    visited_mask[y, x] = 1.0

        return visited_mask

    def get_terrain_at(self, position: Position):
        try:
            y, x = position

            if not (0 <= y < self._terrain_map.shape[0] and 0 <= x < self._terrain_map.shape[1]):
                raise IndexError(f"Position {position} is out of map boundaries")

            return TerrainType(self._terrain_map[y, x])

        except ValueError as e:
            self._logger.error(f"Invalid position format: {e}")
            raise

    def get_world_map(self):
        return self._world_map

    def get_entities(self, type: EntityType) -> EntityList:
        return [flora for flora in self._world_map.get_entities()
                if flora.get_type() == type]

    def get_entity_by_id(self, id: int) -> Optional[Entity]:
        return self._entity_registry.get(id, None)

    def is_valid_position(self, position: Position, entity_id: Optional[int] = None) -> bool:
        return self._is_valid_position(position, entity_id)
