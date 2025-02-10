from logging import Logger
from typing import Optional, cast

from biome.api.biome_api import BiomeAPI
from config.settings import SimulationSettings, Settings
from simulation.core.bootstrap.bootstrap import Bootstrap
from simulation.core.bootstrap.context.context import Context
from simulation.core.bootstrap.context.context_data import BiomeContextData
from simulation.core.systems.events.dispatcher import EventDispatcher
from utils.loggers import setup_logger


class SimulationEngine:
    def __init__(self, settings: Settings):
        self._context: Optional[Context] = None
        # TODO: El logger tiene que ser cargado por el builder y el contexto
        self._logger: Logger = setup_logger("simulation_engine", "simulation_engine.log")
        try:
            bootstrap: Bootstrap = Bootstrap(settings)
            self._context = bootstrap.get_context()
            biome_context = cast(BiomeContextData, self._context.get("biome_ctx"))
            self.biome_api = BiomeAPI(biome_context)
            EventDispatcher.dispatch("biome_loaded", biome_context.map)
        except Exception as e:
            self._logger = setup_logger("bootstrap", "bootstrap.log")
            self._logger.exception(f"[Simulation Engine] There was an error bootstraping: {e}")

    def run(self):
        self._logger.info("Running simulation...")
