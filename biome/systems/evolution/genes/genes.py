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
Abstract base class for genetic information management.

Defines the interface for gene encoding and component conversion;
provides common gene extraction functionality for entity components.
Ensures genetic information integrity through validation mechanisms -
acts as foundation for specialized flora and fauna gene implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from biome.components.environmental.weather_adaptation import WeatherAdaptationComponent
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.vital import VitalComponent
from biome.entities.entity import Entity
from shared.enums.enums import ComponentType, EntityType
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class Genes(ABC):

    @abstractmethod
    def convert_genes_to_components(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def validate_genes(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def lifespan(self) -> float:
        raise NotImplementedError

def extract_common_genes(entity: Entity, genes):
    genes.lifespan = entity.lifespan

    growth_component: GrowthComponent = entity.get_component(ComponentType.GROWTH)
    if growth_component:
        genes.growth_modifier = growth_component.growth_modifier
        genes.growth_efficiency = growth_component.growth_efficiency
        genes.max_size = growth_component.max_size

    vital_component: VitalComponent = entity.get_component(ComponentType.VITAL)
    if vital_component:
        genes.max_vitality = vital_component.max_vitality
        genes.aging_rate = vital_component.aging_rate
        genes.health_modifier = vital_component.health_modifier

    weather_adaptation: WeatherAdaptationComponent = entity.get_component(ComponentType.WEATHER_ADAPTATION)
    if weather_adaptation:
        genes.cold_resistance = weather_adaptation.cold_resistance
        genes.heat_resistance = weather_adaptation.heat_resistance
        genes.optimal_temperature = weather_adaptation.optimal_temperature




