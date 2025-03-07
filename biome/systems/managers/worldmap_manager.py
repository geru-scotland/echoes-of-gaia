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
import traceback
from logging import Logger

from simpy import Environment as simpyEnv

from biome.entities.entity import Entity
from biome.systems.maps.spawn_system import SpawnSystem
from biome.systems.maps.worldmap import WorldMap
from shared.enums.strings import Loggers
from shared.types import TileMap, EntityList, EntityDefinitions, EntityRegistry, \
    TerrainMap, EntityIndexMap
from utils.loggers import LoggerManager


class WorldMapManager:

    def __init__(self, env: simpyEnv, tile_map: TileMap, flora_definitions: EntityDefinitions,
                 fauna_definitions: EntityDefinitions):
        # Quizá worldmapmanager
        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._logger.info("Creating Worldmap...")
        self._env: simpyEnv = env
        self._terrain_map: TerrainMap = tile_map

        try:
            self._spawn_system = SpawnSystem(env, tile_map)
            self._spawn_system.spawn(flora_definitions, fauna_definitions)
            # quitar habitatds de worldmap, no compete al mapa tener las especificaciones
            # quizá, como mucho, habitat cache
            self._entity_registry: EntityRegistry = self._spawn_system.get_entity_registry()
            self._entity_index_map: EntityIndexMap = self._spawn_system.get_entity_index_map()
            self._world_map = WorldMap(tile_map=tile_map, entity_registry=self._entity_registry, entity_index_map=self._entity_index_map)
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.error("There was an exception spawning entities: %s", tb)

    def add_entity(self, entity: Entity):
        pass

    def remove_entity(self, entity: Entity):
        pass

    def get_world_map(self):
        return self._world_map

    def get_entities(self) -> EntityList:
        return self._world_map.get_entities()

    def _is_valid_position(self):
        pass
