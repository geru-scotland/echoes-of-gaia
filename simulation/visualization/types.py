""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""

"""
Type definitions for visualization data structures and interfaces.

Defines TypedDict classes for snapshot data, entity information,
terrain configuration and viewer settings; provides type safety
for visualization components and ensures consistent data formats
across the visualization pipeline and snapshot handling.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import pygame

from shared.enums.enums import BiomeType, DietType, Season

Point = Tuple[int, int]
Size = Tuple[int, int]
Rect = pygame.Rect

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]

RenderCallback = Callable[[pygame.Surface], None]
EventCallback = Callable[[pygame.event.Event], bool]

TerrainSprite = pygame.Surface
EntitySprite = pygame.Surface


class TerrainTypeInfo(TypedDict):
    id: int
    name: str
    color: Color


class SnapshotTimeInfo(TypedDict):
    raw_ticks: int
    month: int
    year: int
    total_months: int


class TerrainMapData(TypedDict):
    terrain_map: List[List[int]]
    terrain_types: Dict[str, str]
    map_dimensions: Tuple[int, int]


class EntityComponentData(TypedDict):
    transform: Optional[Dict[str, Any]]


class EntityStateFields(TypedDict):
    toxicity: Optional[float]
    size: Optional[float]


class EntityData(TypedDict):
    id: int
    type: str
    species: str
    habitats: List[str]
    state_fields: EntityStateFields
    is_dead: bool
    components: Dict[str, Any]
    evolution_cycle: int
    diet_type: Optional[DietType]


class BiomeScoreContributorData(TypedDict):
    population_balance: float
    toxicity: float
    climate: float
    biodiversity: float
    ecosystem_health: float


class BiomeScoreData(TypedDict):
    score: float
    normalized_score: float
    quality: str
    contributor_scores: BiomeScoreContributorData


class MetricsData(TypedDict):
    avg_toxicity: float
    avg_size: float
    num_flora: int
    num_fauna: int
    avg_stress: float
    evolution_trends: Any
    entity_balance: Any
    climate_adaptation: Any


class SnapshotData(TypedDict):
    snapshot_id: str
    simulation_time: SnapshotTimeInfo
    biome_type: BiomeType
    current_season: Season
    climate_averages: Dict[str, float]
    climate_analysis: Any
    creation_timestamp: int
    terrain: TerrainMapData
    entities: Dict[str, EntityData]
    metrics: MetricsData
    biome_score: BiomeScoreData


class ViewerConfig(TypedDict):
    cell_size: int
    panel_width: int
    window_size: Tuple[int, int]
    fps: int
    font_size: int
    title: str
    background_color: Tuple[int, int, int]
    grid_color: Tuple[int, int, int]
    terrain_colors: Dict[int, Tuple[int, int, int]]
    entity_colors: Dict[str, Tuple[int, int, int]]
    navigation_button_size: Tuple[int, int]
    snapshot_path: str
    use_terrain_sprites: bool
