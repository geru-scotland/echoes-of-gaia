import logging
from abc import ABC
from dataclasses import dataclass

from biome.systems.maps.procedural_maps import Map
from config.settings import Config

@dataclass
class ContextData(ABC):
    pass

@dataclass
class BiomeContextData(ContextData):
    logger: logging.Logger
    map: Map
    config: Config

@dataclass
class SimulationContextData(ContextData):
    logger: logging.Logger
    config: Config
