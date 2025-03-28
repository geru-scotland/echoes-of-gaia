"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from logging import getLogger, Logger
from typing import Optional, Type

from config.settings import Settings
from shared.enums.enums import SimulationMode
from shared.enums.strings import Loggers
from simulation.core.engine import SimulationEngine


class SimulationAPI:
    def __init__(self, settings=Type[Settings], mode: SimulationMode = SimulationMode.NORMAL):
        self._engine: Optional[SimulationEngine] = None
        self._settings: Settings = settings
        self._logger: Logger = getLogger(Loggers.SIMULATION)
        self._simulation_mode: SimulationMode = mode

    def _initialise(self):
        self._engine = SimulationEngine(settings=self._settings)

    def run(self):
        self._initialise()
        if not self._engine:
            self._logger.critical("[CRITICAL] Simulation can not be launched, engine failed.")
        self._engine.run()

    def initialise_training(self):
        self._logger.info("Initialising Simulation on training mode...")
        self._initialise()
        if not self._engine:
            self._logger.critical("[CRITICAL] Simulation can not be launched, engine failed.")

    def finish_training(self):
        if self._engine:
            self._engine.shutdown_training()

    def step(self, time_delta: int = 1) -> int:
        return self._engine.step(time_delta)

    def get_biome(self):
        return self._engine.get_biome()

    def get_simulation_time(self) -> int:
        if hasattr(self._engine, '_env'):
            return self._engine._env.now
        return 0
