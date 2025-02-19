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

from config.settings import Config
from shared.types import Spawns, TileMap


@dataclass
class ContextData(ABC):
    logger_name: str

@dataclass
class BiomeContextData(ContextData):
    tile_map: TileMap
    config: Config
    flora_spawns: Spawns
    fauna_spawns: Spawns

@dataclass
class SimulationContextData(ContextData):
    config: Config
