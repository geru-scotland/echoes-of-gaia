import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Type

from perlin_noise import PerlinNoise

from shared.enums import TerrainType
from shared.constants import TERRAIN_TYPES
from shared.constants import MAP_DEFAULT_SIZE
from utils.exceptions import MapGenerationError

@dataclass
class Map:
    size: Tuple[int, int] = MAP_DEFAULT_SIZE,
    weights: List[int] = field(default_factory=list)
    tile_map: List[List[TerrainType]] = field(default_factory=list)
    noise_map: List[List[float]] = field(default_factory=list)

    def __post_init__(self):
        logger = logging.getLogger("bootstrap")
        logger.info(f"[Map] Initialised with size={self.size} and weights={self.weights}")

class ProceduralMethod(ABC):
    def __init__(self, map: Map, seed: int ):
        self._map = map
        self._seed = seed

    @abstractmethod
    def _normalize_coordinates(self, coordinates: Tuple[int, int], size: Tuple[int, int]) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def _generate_noisemap(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _generate_tilemap(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def tile_map(self):
        raise NotImplementedError

    def generate(self) -> Map:
        self._generate_tilemap()
        return self._map


class PerlinNoiseGenerator(ProceduralMethod):
    """
    Código inspirado en:
    https://github.com/CodingQuest2023/Algorithms
    """
    def __init__(self, map: Map, seed: int):
        super().__init__(map, seed)
        self._generate_noisemap()


    def _normalize_coordinates(self, coordinates: Tuple[int, int], size: Tuple[int, int]) -> List[float]:
        x, y = coordinates
        width, height = size
        return [x / width, y / height]

    def _generate_noisemap(self) -> None:
        perlin_3_freq = PerlinNoise(octaves=3, seed=self._seed)
        perlin_6_freq = PerlinNoise(octaves=6, seed=self._seed)
        perlin_12_freq = PerlinNoise(octaves=12, seed=self._seed)
        perlin_24_freq = PerlinNoise(octaves=24, seed=self._seed)

        grid_width, grid_height = self._map.size
        grid_width += 1
        grid_height += 1

        self._map.noise_map =  [ [0.0 for x in range(grid_width)] for y in range(grid_height)]

        for y in range(grid_height):
            for x in range(grid_width):
                normalized_coordinates: List[float] = self._normalize_coordinates((x,y), (grid_width, grid_height))
                noise_value = perlin_3_freq(normalized_coordinates)
                noise_value += 0.5 * perlin_6_freq(normalized_coordinates)
                noise_value += 0.25 * perlin_12_freq(normalized_coordinates)
                noise_value += 0.125 * perlin_24_freq(normalized_coordinates)
                self._map.noise_map[y][x] = noise_value

    def _generate_tilemap(self):
        total_weights: int = sum(self._map.weights)
        vector: List[float] = [ item for row in self._map.noise_map for item in row ]
        min_value: float = min(vector)
        max_value: float = max(vector)
        total_range: float = max_value - min_value

        max_terrain_heights: List[float] = []
        previous_range_height: float = min_value
        for terrain_type in TERRAIN_TYPES:
            height = total_range * (self._map.weights[int(terrain_type)] / total_weights) + previous_range_height
            max_terrain_heights.append(height)
            previous_range_height = height

        max_terrain_heights[int(TerrainType.SNOW)] = max_value

        self._map.tile_map = [ [ TerrainType.GRASS for item in row] for row in self._map.noise_map ]
        for y in range(len(self._map.tile_map)):
            for x in range(len(self._map.tile_map[0])):
                for terrain_type in TERRAIN_TYPES:
                    if self._map.noise_map[y][x] <= max_terrain_heights[int(terrain_type)]:
                        self._map.tile_map[y][x] = terrain_type
                        break
    @property
    def tile_map(self):
        return self._map.tile_map


class MapGenerator:
    def __init__(self, algorithm: Type[ProceduralMethod]):
        self._algorithm = algorithm
        self._logger = logging.getLogger("bootstrap")
        self._logger.info("[MapGenerator] Initialising Map generator")

    def generate(self, map_data: Dict[str, Any], seed: int = 3) -> Map:
        try:
            self._logger.info("[MapGenerator] Generating new map...")
            map: Map = Map(**map_data)
            perlin = self._algorithm(map=map, seed=seed)
            return perlin.generate()
        except Exception as e:
            raise MapGenerationError(f"{e}")