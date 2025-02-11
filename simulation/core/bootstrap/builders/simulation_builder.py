
from logging import Logger
from typing import Optional

from config.settings import Config, SimulationSettings
from simulation.core.bootstrap.context.context_data import SimulationContextData
from simulation.core.bootstrap.builders.builder import Builder


class SimulationBuilder(Builder):
    def __init__(self, settings: SimulationSettings, logger: Logger):
        super().__init__(logger)
        self._settings = settings
        self._logger.info("[Simulation Builder] Initialising SimulationBuilder...")
        self._context_data: Optional[SimulationContextData] = None
        self._initialise()

    def _initialise(self):
        pass

    def build(self) -> None:
        self._logger.info("[Simulation Builder] Simulation builder...")
        try:
            logger = self._settings.get_logger()
            config: Config = self._settings.config.get("simulation")
            self._context = SimulationContextData(config=config, logger=logger)
        except Exception as e:
            self._logger.exception(f"There was a problem building the context from the Simulation: {e}")