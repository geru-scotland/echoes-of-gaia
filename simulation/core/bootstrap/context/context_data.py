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
import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any

from biome.systems.maps.procedural_maps import Map
from config.settings import Config
from shared.types import Spawns


@dataclass
class ContextData(ABC):
    logger: logging.Logger

@dataclass
class BiomeContextData(ContextData):
    map: Map
    config: Config
    flora_spawns: Spawns
    fauna_spawns: Spawns

@dataclass
class SimulationContextData(ContextData):
    config: Config
