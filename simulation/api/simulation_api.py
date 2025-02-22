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
import sys
from logging import getLogger, Logger
from typing import Optional, Type

from config.settings import Settings
from shared.strings import Loggers
from simulation.core.engine import SimulationEngine


class SimulationAPI:
    def __init__(self, settings = Type[Settings]):
        self._engine: Optional[SimulationEngine] = None
        self._settings: Settings = settings
        self._logger: Logger = getLogger(Loggers.SIMULATION)

    def initialise(self):
        self._engine = SimulationEngine(settings=self._settings)

    def run(self):
        self.initialise()
        if not self._engine:
            self._logger.critical("[CRITICAL] Simulation can not be launched, engine failed.")
            # sys.exit(1)
        self._engine.run()
