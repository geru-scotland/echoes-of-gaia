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
import logging
import random
from logging import Logger
from typing import Optional, Dict, Any, Tuple

import numpy as np

from config.settings import BiomeSettings, Config
from biome.systems.maps.procedural_maps import MapGenerator, MapGenData, PerlinNoiseGenerator
from exceptions.custom import MapGenerationError
from shared.constants import MAP_DEFAULT_SIZE
from shared.enums import BiomeType
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from shared.types import EntityDefinitions, TileMap
from simulation.core.bootstrap.context.context_data import BiomeContextData
from simulation.core.bootstrap.builders.builder import Builder, ConfiguratorStrategy
from utils.loggers import LoggerManager
from utils.middleware import log_execution_time


class MapConfigurator(ConfiguratorStrategy):
    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._map: Optional[MapGenData] = None

    @log_execution_time("Map Generation")
    def configure(self, settings: BiomeSettings, **kwargs: Any) -> None:
        map_size: Tuple[int, int]
        config: Config = kwargs.get("config")

        try:
            map_size = config.get("map").get("size")
        except:
            map_size = MAP_DEFAULT_SIZE

        biomes = BiomeStore.biomes
        map_data: Dict[str, Any] = {
            "size": map_size,
            "weights": np.array(biomes.get(config.get("type", {}), BiomeType.TROPICAL).get("weights", []))
        }

        try:
            self._map = MapGenerator(PerlinNoiseGenerator).generate(map_data=map_data, seed=random.randint(1, 99))
            self._logger.debug(self._map.tile_map)
        except MapGenerationError as e:
            logging.error(f"[ERROR] Map generation failed: {e}")
            self._map = None
            raise

    def get_map_gen_data(self) -> MapGenData:
        return self._map

    def get_tile_map(self) -> TileMap:
        return self._map.tile_map

class BiomeBuilder(Builder):
    """
    Cargar settings, inicializar sistemas de eventosGra
    """
    def __init__(self, settings: BiomeSettings):
        super().__init__()
        self._settings = settings
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("[Biome Builder] Initialising BiomeBuilder...")
        self._context_data: Optional[BiomeContextData] = None
        self._initialise()

    def _initialise(self):
        pass

    def build(self) -> None:
        self._logger.info("[Biome Builder] Building biome...")
        # Logs, settings, maps, init events
        try:
            logger = self._settings.get_logger()
            config: Config = self._settings.config.get("biome")
            flora: EntityDefinitions = config.get("flora", {})
            fauna: EntityDefinitions = config.get("fauna", {})
            map_configurator: MapConfigurator = MapConfigurator()
            map_configurator.configure(self._settings, config=config)
            self._context = BiomeContextData(tile_map=map_configurator.get_tile_map(),
                                             config=config, logger_name=Loggers.BIOME,
                                             flora_definitions=flora, fauna_definitions=fauna)
        except Exception as e:
            self._logger.exception(f"There was a problem building the context from the Biome: {e}")