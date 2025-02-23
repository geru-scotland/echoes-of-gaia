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
from typing import List, Tuple, Dict, TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray
from pygame import Surface

from shared.enums import TerrainType, ComponentType, Habitats

if TYPE_CHECKING:
    from biome.entities.entity import Entity
    from biome.components.component import Component
# Tipos, me acabo de enterar que puedo definir tipos custom en Python,
# algo parecido al typedef de c++; y soy un poco más feliz:

# Coordenadas y mapeoss de terrenos
Coords = Tuple[int, int]
TileMappings = Dict[TerrainType, List[Coords]]
TerrainSpritesMapping = Dict[TerrainType, List[Surface]]

# Listas
EntityList = List["Entity"]
ComponentDict = Dict["ComponentType", "Component"]
TerrainList = np.ndarray
EntityDefinitions = List[Dict[str, Any]]
ComponentData = Dict[str, Any]
HabitatList = List[Habitats.Type]

# Mapas
TileMap = np.ndarray
NoiseMap = np.ndarray
EntityLayer = List[EntityList]
TerrainMap = np.ndarray
EntityIndexMap = np.ndarray
EntityRegistry = Dict[int, "Entity"]

BiomeStoreData = Dict[str, Any]
HabitatCache = Dict[Habitats.Type, ndarray]
BiomeStateData = Dict[str, Any]