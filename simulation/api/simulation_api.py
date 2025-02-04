from logging import Logger
from typing import Optional

from simulation.engine import SimulationEngine
from utils.loggers import setup_logger


class SimulationAPI:
    def __init__(self):
        self._engine: Optional[SimulationEngine] = None

    def initialise(self):
        self._engine = SimulationEngine()

    def run(self):
        self._engine.run()
