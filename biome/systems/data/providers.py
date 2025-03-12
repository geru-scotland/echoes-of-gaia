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
from abc import ABC, abstractmethod

from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector


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