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


def fauna_genes_to_individual(fauna_genes: FaunaGenes):
    fauna_genes.validate_genes()

    return creator.Individual([
        (fauna_genes.growth_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (fauna_genes.growth_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (fauna_genes.max_size - 0.1) / 4.9,  # De 0.1-5.0 a 0-1
        (fauna_genes.max_vitality - 50.0) / 150.0,  # De 50-200 a 0-1
        (fauna_genes.aging_rate - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (fauna_genes.health_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (fauna_genes.max_energy_reserves - 50.0) / 100.0,  # De 50-150 a 0-1
        (fauna_genes.lifespan - 1.0) / 999.0,  # De 1-1000 a 0-1

        (fauna_genes.cold_resistance - 0.01) / 0.99,  # 0-1
        (fauna_genes.heat_resistance - 0.01) / 0.99,  # 0-1

        (fauna_genes.optimal_temperature + 20.0) / 60.0,  # De -20-40 a [0,1]

        # TODO: Especificos de fauna cuando  haga
        # (fauna_genes.foraging_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        # fauna_genes.predator_avoidance,  # De 0.0-1.0 a 0-1
    ])


def deap_genes_to_fauna_genes(individual) -> FaunaGenes:
    genes = FaunaGenes()

    clamped_individual = [max(0.0, min(1.0, val)) for val in individual]

    genes.growth_modifier = 0.1 + (clamped_individual[0] * 1.9)  # 0.1-2.0
    genes.growth_efficiency = 0.1 + (clamped_individual[1] * 0.9)  # 0.1-1.0
    genes.max_size = 0.1 + (clamped_individual[2] * 4.9)  # 0.1-5.0
    genes.max_vitality = 50.0 + (clamped_individual[3] * 150.0)  # 50-200
    genes.aging_rate = 0.1 + (clamped_individual[4] * 1.9)  # 0.1-2.0
    genes.health_modifier = 0.1 + (clamped_individual[5] * 1.9)  # 0.1-2.0
    genes.max_energy_reserves = 50.0 + (clamped_individual[6] * 100.0)  # 50-150
    genes.lifespan = 1.0 + (clamped_individual[7] * 999.0)  # 1-1000 años

    genes.cold_resistance = 0.01 + clamped_individual[8] * 0.99  # 0-1
    genes.heat_resistance = 0.01 + clamped_individual[9] * 0.99  # 0-1

    genes.optimal_temperature = -20.0 + (clamped_individual[10] * 60.0)  # -20 a 40

    # TODO: Especificos de fauna cuando  haga
    # genes.foraging_efficiency = 0.1 + (clamped_individual[12] * 0.9)  # 0.1-1.0
    # genes.predator_avoidance = clamped_individual[13]  # 0.0-1.0

    return genes

def extract_genes_from_fauna(fauna_entity: Entity) -> FaunaGenes:
    genes = FaunaGenes()
    extract_common_genes(fauna_entity, genes)
    # TODO: Específicos de fauna, cuando los tenga
    return genes