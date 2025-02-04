from logging import Logger
from typing import Optional

from config.settings import Settings, BiomeSettings, Config
from biome.systems.maps.procedural_maps import MapGenerator, Map
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
            map: Map = MapGenerator().generate()
            logger = self._biome_settings.get_logger()
            config: Config = self._biome_settings.get_config()
            return BiomeContextData(map=map, config=config, logger=logger)
        except:
            self._logger.exception("There was a problem building the context from the Biome")