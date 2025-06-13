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

"""
Dataclass definitions for biome and simulation context structures.

Defines BiomeContextData with terrain, entities and climate models;
SimulationContextData includes telemetry and configuration data.
Provides typed containers for bootstrap context management.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from config.settings import Config
from shared.enums.enums import BiomeType
from shared.types import TileMap, EntityDefinitions
from simulation.core.systems.telemetry.manager import InfluxDBManager


@dataclass
class ContextData(ABC):
    logger_name: str


@dataclass
class BiomeContextData(ContextData):
    biome_type: BiomeType
    tile_map: TileMap
    config: Config
    flora_definitions: EntityDefinitions
    fauna_definitions: EntityDefinitions
    climate_model: str
    fauna_model: str
    fauna_ia: bool


@dataclass
class SimulationContextData(ContextData):
    config: Config
    influxdb: Optional[InfluxDBManager]
