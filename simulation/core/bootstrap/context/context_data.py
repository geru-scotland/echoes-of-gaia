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
from shared.types import TileMap, EntityDefinitions


@dataclass
class ContextData(ABC):
    logger_name: str

@dataclass
class BiomeContextData(ContextData):
    tile_map: TileMap
    config: Config
    flora_definitions: EntityDefinitions
    fauna_definitions: EntityDefinitions

@dataclass
class SimulationContextData(ContextData):
    config: Config
