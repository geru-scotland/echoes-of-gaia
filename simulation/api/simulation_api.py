from typing import Optional

from simulation.engine import SimulationEngine

class SimulationAPI:
    def __init__(self):
        self._engine: Optional[SimulationEngine] = None

    def initialise(self):
        self._engine = SimulationEngine()

    def run(self):
        self._engine.run()
