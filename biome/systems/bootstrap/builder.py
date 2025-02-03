from logging import Logger
from typing import Any, Optional, Tuple

from config.settings import Settings, BiomeSettings, Config
from utils.loggers import setup_logger
from biome.systems.maps.procedural_maps import MapGenerator, Map
from biome.systems.context.manager import ContextManager, Context

class BiomeBuilder:
    """
    Cargar settings, inicializar sistemas de eventosGra
    """
    def __init__(self, **kwargs: Any):
        self._logger = setup_logger("bootstrap", "bootstrap.log")
        self._logger.info("[Biome Builder] Initialising BiomeBuilder...")
        self._context_manager: Optional[ContextManager] = None
        self._biome_settings: Optional[BiomeSettings] = None
        self._initialise()

    def _initialise(self):
        self._context_manager: ContextManager = ContextManager()
        self._biome_settings = Settings().biome_settings

    def build(self) -> Tuple[Context, Logger]:
        self._logger.info("[Biome Builder] Building biome...")
        # Logs, settings, maps, init events
        map: Map = MapGenerator().generate()
        logger = self._biome_settings.get_logger()
        config: Config = self._biome_settings.get_config()
        # build_context ha de recibir settings + map info
        context: Context = self._context_manager.build_context(map=map,
                                                               config=config)
        # devolveré una tupla con logger y contexto
        return context, logger