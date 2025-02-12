import logging
from abc import ABC
from dataclasses import dataclass

from biome.systems.maps.procedural_maps import Map
from config.settings import Config

@dataclass
class ContextData(ABC):
    logger: logging.Logger

@dataclass
class BiomeContextData(ContextData):
    map: Map
    config: Config

@dataclass
class SimulationContextData(ContextData):
    config: Config
