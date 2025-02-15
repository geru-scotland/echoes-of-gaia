from typing import List, Tuple, Dict, TYPE_CHECKING, Any

import numpy as np
from pygame import Surface

from shared.enums import TerrainType, ComponentType

if TYPE_CHECKING:
    from biome.entities.entity import Entity
    from biome.components.component import Component
# Tipos, me acabo de enterar que puedo definir tipos custom en Python,
# algo parecido al typedef de c++; y soy un poco más feliz:

# Mapas
TileMap = np.ndarray
NoiseMap = np.ndarray


# Coordenadas y mapeoss de terrenos
Coords = Tuple[int, int]
TileMappings = Dict[TerrainType, List[Coords]]
TerrainSpritesMapping = Dict[TerrainType, List[Surface]]

# Listas
EntityList = List["Entity"]
ComponentDict = Dict["ComponentType", "Component"]
TerrainList = np.ndarray
Spawns = List[Dict[str, Any]]

ComponentData = Dict[str, Any]