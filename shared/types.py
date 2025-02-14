from typing import List, Tuple, Dict, TYPE_CHECKING, Any

from pygame import Surface

from shared.enums import TerrainType, ComponentType

if TYPE_CHECKING:
    from biome.entities.entity import Entity
    from biome.components.component import Component
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
EntityList = List["Entity"]
ComponentDict = Dict["ComponentType", "Component"]
TerrainList = List["TerrainType"]
Spawns = List[Dict[str, Any]]

ComponentData = Dict[str, Any]