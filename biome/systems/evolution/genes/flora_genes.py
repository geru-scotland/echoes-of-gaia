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
from deap import creator

from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from biome.components.physiological.photosynthetic_metabolism import PhotosyntheticMetabolismComponent
from biome.entities.entity import Entity
from biome.systems.evolution.genes.genes import Genes, extract_common_genes
from typing import List, Dict, Any

from shared.enums.enums import ComponentType
from shared.evolution.ranges import FLORA_GENE_RANGES

class FloraGenes(Genes):
    def __init__(self):
        self.growth_modifier = 0.0
        self.growth_efficiency = 0.0
        self.max_size = 0.0

        self.max_vitality = 0.0
        self.aging_rate = 0.0
        self.health_modifier = 0.0

        self.base_photosynthesis_efficiency = 0.0
        self.base_respiration_rate = 0.0
        self._lifespan = 0.0
        self.metabolic_activity = 0.0
        self.max_energy_reserves = 0.0

        self.cold_resistance = 0.0
        self.heat_resistance = 0.0
        self.optimal_temperature = 0.0

        self.nutrient_absorption_rate = 0.0
        self.mycorrhizal_rate = 0.0
        self.base_nutritive_value = 0.0
        self.base_toxicity = 0.0
        # Dejo las siguientes para más adelante:
        # self.drought_resistance = 0.0
        # self.toxicity = 0.0

    @property
    def lifespan(self) -> float:
        return self._lifespan

    @lifespan.setter
    def lifespan(self, value: float):
        self._lifespan = value

    def __str__(self):
        fields = {
            "growth_modifier": self.growth_modifier,
            "growth_efficiency": self.growth_efficiency,
            "max_size": self.max_size,
            "max_vitality": self.max_vitality,
            "aging_rate": self.aging_rate,
            "health_modifier": self.health_modifier,
            "base_photosynthesis_efficiency": self.base_photosynthesis_efficiency,
            "base_respiration_rate": self.base_respiration_rate,
            "lifespan": self.lifespan,
            "metabolic_activity": self.metabolic_activity,
            "max_energy_reserves": self.max_energy_reserves,
            "cold_resistance": self.cold_resistance,
            "heat_resistance": self.heat_resistance,
            "nutrient_absorption_rate": self.nutrient_absorption_rate,
            "mycorrhizal_rate": self.mycorrhizal_rate,
            "base_nutritional_value": self.base_nutritive_value,
            "base_toxicity": self.base_toxicity
        }
        return ", ".join(f"{key}={value}" for key, value in fields.items())

    def convert_genes_to_components(self) -> List[Dict[str, Any]]:
        components: List[Dict[str, Any]] = []

        growth_component: Dict[str, Any] = {
            "GrowthComponent": {
                "growth_modifier": self.growth_modifier,
                "growth_efficiency": self.growth_efficiency,
                "lifespan": self.lifespan,
                "max_size": self.max_size * 5.0
            }
        }
        components.append(growth_component)

        vital_component: Dict[str, Any] = {
            "VitalComponent": {
                "max_vitality": self.max_vitality,
                "aging_rate": self.aging_rate,
                "lifespan": self.lifespan,
                "health_modifier": self.health_modifier,
            }
        }
        components.append(vital_component)

        metabolic_component: Dict[str, Any] = {
            "PhotosyntheticMetabolismComponent": {
                "photosynthesis_efficiency": self.base_photosynthesis_efficiency,
                "respiration_rate": self.base_respiration_rate,
                "metabolic_activity": self.metabolic_activity,
                "max_energy_reserves": self.max_energy_reserves,
            }
        }
        components.append(metabolic_component)

        nutritional_component: Dict[str, Any] = {
            "AutotrophicNutritionComponent": {
                "nutrient_absorption_rate": self.nutrient_absorption_rate,
                "mycorrhizal_rate": self.mycorrhizal_rate,
                "base_nutritive_value": self.base_nutritive_value,
                "base_toxicity": self.base_toxicity,
                "max_energy_reserves": self.max_energy_reserves
            }
        }
        components.append(nutritional_component)

        weather_adaptation_component: Dict[str, Any] = {
            "WeatherAdaptationComponent": {
                "cold_resistance": self.cold_resistance,
                "heat_resistance": self.heat_resistance,
                "optimal_temperature": self.optimal_temperature,
            }
        }
        components.append(weather_adaptation_component)


        return components

    def validate_genes(self) -> None:

        for attr, (min_val, max_val) in FLORA_GENE_RANGES.items():
            if hasattr(self, attr):
                current_val = getattr(self, attr)
                valid_val = max(min_val, min(current_val, max_val))
                setattr(self, attr, valid_val)

def extract_genes_from_flora(flora_entity: Entity) -> FloraGenes:
    genes = FloraGenes()
    extract_common_genes(flora_entity, genes)

    metabolic_component: PhotosyntheticMetabolismComponent = flora_entity.get_component(
        ComponentType.PHOTOSYNTHETIC_METABOLISM)
    if metabolic_component:
        genes.base_photosynthesis_efficiency = metabolic_component.base_photosynthesis_efficiency
        genes.base_respiration_rate = metabolic_component.base_respiration_rate
        genes.metabolic_activity = metabolic_component.metabolic_activity
        genes.max_energy_reserves = metabolic_component.max_energy_reserves

    nutritional_component: AutotrophicNutritionComponent = flora_entity.get_component(
        ComponentType.AUTOTROPHIC_NUTRITION)
    if nutritional_component:
        genes.nutrient_absorption_rate = nutritional_component.nutrient_absorption_rate
        genes.mycorrhizal_rate = nutritional_component.mycorrhizal_rate
        genes.base_nutritive_value = nutritional_component.base_nutritive_value
        genes.base_toxicity = nutritional_component.base_toxicity

    return genes