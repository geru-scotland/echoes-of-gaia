import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

from shared.enums import TerrainType
from shared.constants import MAP_DEFAULT_SIZE
from utils.exceptions import MapGenerationError

@dataclass
class Map:
    size: Tuple[int, int] = MAP_DEFAULT_SIZE,
    weights: List[int] = field(default_factory=list)
    tile_map: List[TerrainType] = field(default_factory=list)
    noise_map: List[float] = field(default_factory=list)

    def __post_init__(self):
        logger = logging.getLogger("bootstrap")
        logger.info(f"[Map] Initialised with size={self.size} and weights={self.weights}")

class ProceduralMethod(ABC):
    @abstractmethod
    def generate_noisemap(self, size: Tuple[int, int], seed):
        raise NotImplementedError

    @abstractmethod
    def get_tailedmap(self, terrain_weights: List[int]):
        raise NotImplementedError

class PerlinNoiseGenerator(ProceduralMethod):
    def __init__(self, map: Map):
        self._map = map

    def generate_noisemap(self, size: Tuple[int, int], seed):
        pass

    def get_tailedmap(self, terrain_weights: List[int]):
        pass

class MapGenerator:
    def __init__(self):
        self._logger = logging.getLogger("bootstrap")
        self._logger.info("[MapGenerator] Initialising Map generator")

    def generate(self, map_data: Dict[str, Any]) -> Map:
        try:
            self._logger.info("[MapGenerator] Generating new map...")
            map: Map = Map(**map_data)
            perlin: PerlinNoiseGenerator = PerlinNoiseGenerator(map)
        except Exception as e:
            raise MapGenerationError(f"{e}")