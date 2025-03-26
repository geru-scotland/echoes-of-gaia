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
from typing import List, Dict, Any

from deap import creator

from biome.entities.entity import Entity
from biome.systems.evolution.genes.genes import Genes, extract_common_genes
from shared.enums.enums import ComponentType


class FaunaGenes(Genes):
    def __init__(self):
        # Los compartidos con flora
        self.growth_modifier = 0.0
        self.growth_efficiency = 0.0
        self.max_size = 0.0
        self.max_vitality = 0.0
        self.aging_rate = 0.0
        self.health_modifier = 0.0
        self.max_energy_reserves = 0.0
        self._lifespan = 0.0

        # resistencias
        self.cold_resistance = 0.0
        self.heat_resistance = 0.0
        self.optimal_temperature = 0.0

        # aquí pondré genes específicos de fauna
        # self.foraging_efficiency = 0.0
        # self.predator_avoidance = 0.0

    def convert_genes_to_components(self) -> list[dict[str, any]]:
        components: list[dict[str, any]] = []

        # componentes compartidos con flora
        growth_component: dict[str, any] = {
            "GrowthComponent": {
                "growth_modifier": self.growth_modifier,
                "growth_efficiency": self.growth_efficiency,
                "lifespan": self.lifespan,
                "max_size": self.max_size * 5.0
            }
        }
        components.append(growth_component)

        vital_component: dict[str, any] = {
            "VitalComponent": {
                "max_vitality": self.max_vitality,
                "aging_rate": self.aging_rate,
                "lifespan": self.lifespan,
                "health_modifier": self.health_modifier,
            }
        }
        components.append(vital_component)

        weather_adaptation_component: Dict[str, Any] = {
            "WeatherAdaptationComponent": {
                "cold_resistance": self.cold_resistance,
                "heat_resistance": self.heat_resistance,
                "optimal_temperature": self.optimal_temperature,
            }
        }
        components.append(weather_adaptation_component)

        # TODO: Aquí, cuando ya tenga los especificos de fauna.

        return components

    def validate_genes(self) -> None:
        from shared.evolution.ranges import FAUNA_GENE_RANGES

        for attr, (min_val, max_val) in FAUNA_GENE_RANGES.items():
            if hasattr(self, attr):
                current_val = getattr(self, attr)
                valid_val = max(min_val, min(current_val, max_val))
                setattr(self, attr, valid_val)

    @property
    def lifespan(self) -> float:
        return self._lifespan

    @lifespan.setter
    def lifespan(self, value: float):
        self._lifespan = value

def extract_genes_from_fauna(fauna_entity: Entity) -> FaunaGenes:
    genes = FaunaGenes()
    extract_common_genes(fauna_entity, genes)
    # TODO: Específicos de fauna, cuando los tenga
    return genes