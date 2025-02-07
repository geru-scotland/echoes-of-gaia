from logging import Logger
from typing import Optional, Type

from simulation.core.engine import SimulationEngine
from simulation.renderer.renderer import Renderer
from utils.loggers import setup_logger


class SimulationAPI:
    def __init__(self, render: Type[Renderer]):
        self._engine: Optional[SimulationEngine] = None
        render()

    def initialise(self):
        self._engine = SimulationEngine()

    def run(self):
        self._engine.run()
