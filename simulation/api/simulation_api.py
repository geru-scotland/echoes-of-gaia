from logging import Logger
from typing import Optional, Type

from config.settings import SimulationSettings, Settings
from simulation.core.engine import SimulationEngine
from utils.loggers import setup_logger


class SimulationAPI:
    def __init__(self, settings = Type[Settings]):
        self._engine: Optional[SimulationEngine] = None
        self._settings: Settings = settings

    def initialise(self):
        self._engine = SimulationEngine(settings=self._settings)

    def run(self):
        self._engine.run()
