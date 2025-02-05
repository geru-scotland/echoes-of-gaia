from logging import Logger
from typing import Optional, Dict, Any, Tuple

from config.settings import Settings, BiomeSettings, Config
from biome.systems.maps.procedural_maps import MapGenerator, Map
from shared.constants import BIOME_TYPE_WEIGHTS, MAP_DEFAULT_SIZE
from simulation.bootstrap.context.context_data import BiomeContextData

class BiomeBuilder:
    """
    Cargar settings, inicializar sistemas de eventosGra
    """
    def __init__(self, logger: Logger):
        self._logger: Logger = logger
        self._logger.info("[Biome Builder] Initialising BiomeBuilder...")
        self._biome_settings: Optional[BiomeSettings] = None
        self._initialise()

    def _initialise(self):
        self._biome_settings = Settings().biome_settings

    def build(self) -> BiomeContextData:
        self._logger.info("[Biome Builder] Building biome...")
        # Logs, settings, maps, init events
        try:
            logger = self._biome_settings.get_logger()
            config: Config = self._biome_settings.get_config().get("biome")
            map_size: Tuple[int, int]
            try:
                map_size = config.get("map").get("size")
            except:
                map_size = MAP_DEFAULT_SIZE
            map_data: Dict[str, Any] = {
                "size": map_size,
                "weights": BIOME_TYPE_WEIGHTS[config.get("type", {})]
            }
            map: Map = MapGenerator().generate(map_data=map_data)
            return BiomeContextData(map=map, config=config, logger=logger)
        except Exception as e:
            self._logger.exception(f"There was a problem building the context from the Biome: {e}")