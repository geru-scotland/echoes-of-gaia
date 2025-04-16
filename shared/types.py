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
from pathlib import Path
from typing import List, Tuple, Dict, TYPE_CHECKING, Any, Optional, Callable, Union, TypeAlias

import numpy as np
from numpy import ndarray

from biome.components.base.component import Component
from shared.enums.enums import TerrainType, Habitats, EntityType, ComponentType, FaunaSpecies, FaunaAction, Direction

if TYPE_CHECKING:
    pass
# Tipos, me acabo de enterar que puedo definir tipos custom en Python,
# algo parecido al typedef de c++; y soy un poco más feliz:

# Coordenadas y mapeoss de terrenos
Coords: TypeAlias = Tuple[int, int]
TileMappings: TypeAlias = Dict[TerrainType, List[Coords]]
TerrainSpritesMapping: TypeAlias = Dict[TerrainType, List[Any]]

# Listas
EntityList: TypeAlias = List["Entity"]
ComponentDict: TypeAlias = Dict["ComponentType", "Component"]
TerrainList: TypeAlias = np.ndarray
EntityDefinitions: TypeAlias = List[Dict[str, Any]]
ComponentData: TypeAlias = Dict[str, Any]
HabitatList: TypeAlias = List[Habitats.Type]

# Mapas
TileMap: TypeAlias = np.ndarray
NoiseMap: TypeAlias = np.ndarray
EntityLayer: TypeAlias = List[EntityList]
TerrainMap: TypeAlias = np.ndarray
EntityIndexMap: TypeAlias = np.ndarray
EntityRegistry: TypeAlias = Dict[int, "Entity"]
Position: TypeAlias = Tuple[int | ndarray, int | ndarray]
BiomeStoreData: TypeAlias = Dict[str, Any]
HabitatCache: TypeAlias = Dict[Habitats.Type, ndarray]

# Snapshots
SnapshotID: TypeAlias = str
TimeStamp: TypeAlias = int
SimulationTime: TypeAlias = int
EntityData: TypeAlias = Dict[str, Any]
TerrainData: TypeAlias = Dict[str, Any]
BiomeScoreData: TypeAlias = Dict[str, Any]
MetricsData: TypeAlias = Dict[str, Any]
ClimateData: TypeAlias = Dict[str, Any]
PositionData: TypeAlias = Tuple[int, int]
CallbackType: TypeAlias = Optional[Callable[[Optional[Path]], None]]

# Agents
Observation: TypeAlias = Union[Dict[str, Any]]
Target: TypeAlias = Tuple[FaunaSpecies, EntityType, int]
DecodedAction: TypeAlias = Union[FaunaAction, Direction]

# Neurosymbolic

PredictionResult: TypeAlias = Dict[str, Any]
SymbolicResult: TypeAlias = Dict[str, Any]
IntegratedResult: TypeAlias = Dict[str, Any]
InterventionAction: TypeAlias = Dict[str, Any]
