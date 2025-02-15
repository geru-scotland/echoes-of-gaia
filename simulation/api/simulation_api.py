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
from typing import Optional, Type

from config.settings import Settings
from simulation.core.engine import SimulationEngine


class SimulationAPI:
    def __init__(self, settings = Type[Settings]):
        self._engine: Optional[SimulationEngine] = None
        self._settings: Settings = settings

    def initialise(self):
        self._engine = SimulationEngine(settings=self._settings)

    def run(self):
        self.initialise()
        self._engine.run()
