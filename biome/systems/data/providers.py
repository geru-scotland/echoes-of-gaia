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
Abstract biome data provider interface for component access.

Defines standardized methods for accessing biome data components;
includes entity providers, worldmap, collectors and analyzers.
Provides interface contract for biome type and climate manager
access - enables consistent data layer abstraction patterns.
"""

from abc import ABC, abstractmethod

from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from shared.enums.enums import BiomeType


class BiomeDataProvider(ABC):
    @abstractmethod
    def get_entity_provider(self) -> EntityProvider:
        pass

    @abstractmethod
    def get_world_map(self) -> WorldMap:
        pass

    @abstractmethod
    def get_entity_collector(self) -> EntityDataCollector:
        pass

    @abstractmethod
    def get_climate_collector(self) -> ClimateDataCollector:
        pass

    @abstractmethod
    def get_score_analyzer(self) -> BiomeScoreAnalyzer:
        pass

    @abstractmethod
    def get_biome_type(self) -> BiomeType:
        pass

    @abstractmethod
    def get_climate_data_manager(self):
        pass