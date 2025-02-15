import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Type, Callable

import numpy as np
from numpy import ndarray
from perlin_noise import PerlinNoise

from shared.enums import TerrainType
from shared.constants import MAP_DEFAULT_SIZE
from shared.stores.biome_store import BiomeStore
from shared.types import TileMap, NoiseMap, TerrainList
from utils.exceptions import MapGenerationError

@dataclass
class Map:
    size: Tuple[int, int] = MAP_DEFAULT_SIZE,
    weights: ndarray = field(default_factory=list)
    tile_map: TileMap = field(default_factory=list)
    noise_map: NoiseMap = field(default_factory=list)

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

    def _normalize_coordinates(self, coordinates: Tuple[int, int], size: Tuple[int, int]) -> Tuple[float, float]:
        pass

    def _generate_noisemap(self) -> None:
        perlin_3_freq: Callable = PerlinNoise(octaves=3, seed=self._seed)
        perlin_6_freq: Callable = PerlinNoise(octaves=6, seed=self._seed)
        perlin_12_freq: Callable = PerlinNoise(octaves=12, seed=self._seed)
        perlin_24_freq: Callable = PerlinNoise(octaves=24, seed=self._seed)

        grid_width, grid_height = self._map.size
        grid_width += 1
        grid_height += 1

        # Extraigo indices para montar una matriz con coordenadas
        # para poder vectorizar cálculos en lugar de ir recorriendo
        # coordenada a coordenada para pasar las normalizadas a perlin
        y_indices, x_indices = np.indices((grid_height, grid_width))
        coords: ndarray = np.stack([x_indices, y_indices], axis=-1)

        normalized_coords: ndarray = coords / np.array([grid_height, grid_width])

        self._map.noise_map = np.zeros((grid_height, grid_width))

        # No puedo vectorizar, perlin functions no aceptan np arrays
        for y in range(grid_height):
            for x in range(grid_width):
                noise_value = perlin_3_freq(tuple(normalized_coords[y, x]))
                noise_value += 0.5 * perlin_6_freq(tuple(normalized_coords[y, x]))
                noise_value += 0.25 * perlin_12_freq(tuple(normalized_coords[y, x]))
                noise_value += 0.125 * perlin_24_freq(tuple(normalized_coords[y, x]))
                self._map.noise_map[y, x] = noise_value

    def _generate_tilemap(self):
        terrain_types: TerrainList = BiomeStore.terrains
        total_weights: int = np.sum(self._map.weights)
        min_value: float = self._map.noise_map.min()
        max_value: float = self._map.noise_map.max()
        total_range: float = max_value - min_value

        normalized_weights: ndarray = self._map.weights / total_weights
        # empieza en min value y voy sumando acumjulativamente los valores escalados
        # el autor original, acumulaba en iteración, pero como yo vectorizo con numpy
        # utilizo cumsum
        max_terrain_heights = min_value + np.cumsum(total_range * normalized_weights)
        max_terrain_heights[int(TerrainType.SNOW)] = max_value

        # Copio shape del noise map y lo inicializo con grass
        self._map.tile_map = np.full_like(self._map.noise_map, TerrainType.GRASS, dtype=object)


        # Monto np array de shape noisemap, y para cada celda del noisemap selecciono
        # el terreno que lo cubre, es decir, el terreno ha de tener un valor max tal que
        # el valor del noisemap entre en ese rango. Por eso uso searchsorted, selecciona
        # el primer indice que lo cumple.
        terrain_indices: ndarray = np.searchsorted(max_terrain_heights, self._map.noise_map)

        # A cada celda del tile map, le asigno el tipo de terreno que diga su indice
        self._map.tile_map = terrain_types[terrain_indices]
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