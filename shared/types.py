from typing import List, Tuple, Dict

from pygame import Surface

from biome.components.component import Component
from biome.entities.entity import Entity
from shared.enums import TerrainType, ComponentType

# Tipos, me acabo de enterar que puedo definir tipos custom en Python,
# algo parecido al typedef de c++; y soy un poco más feliz:

# Mapas
TileMap = List[List[TerrainType]]
NoiseMap = List[List[float]]

# Coordenadas y mapeoss de terrenos
Coords = Tuple[int, int]
TileMappings = Dict[TerrainType, List[Coords]]
TerrainSpritesMapping = Dict[TerrainType, List[Surface]]

# Listas
EntityList = List[Entity]
ComponentDict = Dict[ComponentType, Component]