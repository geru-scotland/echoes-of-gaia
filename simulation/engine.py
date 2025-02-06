from logging import Logger
from typing import Optional, cast

from biome.api.biome_api import BiomeAPI
from simulation.bootstrap.bootstrap import Bootstrap
from simulation.bootstrap.context.context import Context
from simulation.bootstrap.context.context_data import BiomeContextData
from utils.loggers import setup_logger


class SimulationEngine:
    def __init__(self):
        self._context: Optional[Context] = None
        # TODO: El logger tiene que ser cargado por el builder y el contexto
        self._logger: Logger = setup_logger("simulation_engine", "simulation_engine.log")
        try:
            bootstrap: Bootstrap = Bootstrap()
            self._context = bootstrap.get_context()
            biome_context = cast(BiomeContextData, self._context.get("biome_ctx"))
            self._logger.debug(biome_context)
            self.biome_api = BiomeAPI(biome_context)
        except Exception as e:
            self._logger = setup_logger("bootstrap", "bootstrap.log")
            self._logger.exception(f"[Simulation Engine] There was an error bootstraping: {e}")

    def run(self):
        self._logger.info("Running simulation...")
