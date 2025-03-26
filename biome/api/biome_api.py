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
from typing import Dict, Any

import simpy

from biome.biome import Biome
from simulation.core.bootstrap.context.context_data import BiomeContextData
from simulation.core.systems.telemetry.datapoint import Datapoint


class BiomeAPI:
    def __init__(self, context: BiomeContextData, env: simpy.Environment, options: Dict[str, Any] = None):
       self.biome = Biome(context, env, options)

    def update(self, era: int, step: int):
        pass