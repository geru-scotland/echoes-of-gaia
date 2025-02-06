import logging
from logging import Logger
from typing import Optional, Dict, Any, Tuple

from config.settings import BiomeSettings, Config, Settings
from biome.systems.maps.procedural_maps import MapGenerator, Map
from shared.constants import BIOME_TYPE_WEIGHTS, MAP_DEFAULT_SIZE
from simulation.bootstrap.context.context_data import BiomeContextData
from simulation.bootstrap.builders.builder import Builder, ConfiguratorStrategy
from utils.exceptions import MapGenerationError


class MapConfigurator(ConfiguratorStrategy):
    def __init__(self):
        self._logger = logging.getLogger()
        self._map: Optional[Map] = None

    def configure(self, settings: BiomeSettings, **kwargs: Any) -> None:
        map_size: Tuple[int, int]
        config: Config = kwargs.get("config")
        try:
            map_size = config.get("map").get("size")
        except:
            map_size = MAP_DEFAULT_SIZE
        map_data: Dict[str, Any] = {
            "size": map_size,
            "weights": BIOME_TYPE_WEIGHTS[config.get("type", {})]
        }
        try:
            self._map = MapGenerator().generate(map_data=map_data)
        except Exception as e:
            raise

    def get_map(self) -> Map:
        return self._map

class BiomeBuilder(Builder):
    """
    Cargar settings, inicializar sistemas de eventosGra
    """
    def __init__(self, settings: BiomeSettings, logger: Logger):
        super().__init__(logger)
        self._settings = settings
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
            map_configurator: MapConfigurator = MapConfigurator()
            map_configurator.configure(self._settings, config=config)
            self._context = BiomeContextData(map=map_configurator.get_map(), config=config, logger=logger)
        except Exception as e:
            self._logger.exception(f"There was a problem building the context from the Biome: {e}")