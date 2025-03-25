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
import random
import sys
from logging import Logger

from biome.systems.evolution.genes.fauna_genes import deap_genes_to_fauna_genes, fauna_genes_to_individual
from biome.systems.evolution.genes.flora_genes import deap_genes_to_flora_genes, flora_genes_to_individual
from shared.enums.enums import EntityType
from shared.enums.strings import Loggers
from shared.evolution.ranges import FAUNA_GENE_RANGES, FLORA_GENE_RANGES
from utils.loggers import LoggerManager


class AdaptiveMutationOperator:
    def __init__(self, indpb: float = 0.1):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.indpb = indpb
        self.entity_type = None
        self.avg_lifespan = 10.0
        self.max_change_percent = 0.1
        self.min_absolute_change = 0.01

    def adapt(self, entity_type: EntityType, avg_lifespan: float) -> None:
        self.entity_type = entity_type
        self.avg_lifespan = avg_lifespan

        max_percent = 0.10
        min_percent = 0.01
        # Pongo 40, pero es un poco por los datos actuales para pruebas, cambiar esto.
        scale = 40.0

        # Cuanto mayor es avg_lifespan, más cerca del maxpercent; cuanto menor, más cerca del 1%
        ratio = min(1.0, avg_lifespan / scale)
        self.max_change_percent = min_percent + (max_percent - min_percent) * ratio

    def __call__(self, individual, indpb=None):
        indpb = indpb if indpb is not None else self.indpb
        entity_type = self.entity_type

        if entity_type is None:
            entity_type = (EntityType.FLORA
                           if len(individual) == len(FLORA_GENE_RANGES)
                           else EntityType.FAUNA)

        if entity_type == EntityType.FLORA:
            genes = deap_genes_to_flora_genes(individual)
            gene_ranges = FLORA_GENE_RANGES
            attr_names = [
                "growth_modifier", "growth_efficiency", "max_size", "max_vitality",
                "aging_rate", "health_modifier", "base_photosynthesis_efficiency",
                "base_respiration_rate", "lifespan", "metabolic_activity",
                "max_energy_reserves", "cold_resistance", "heat_resistance",
                "optimal_temperature", "nutrient_absorption_rate", "mycorrhizal_rate",
                "base_nutritive_value", "base_toxicity"
            ]
        else:  # FAUNA
            genes = deap_genes_to_fauna_genes(individual)
            gene_ranges = FAUNA_GENE_RANGES
            attr_names = [
                "growth_modifier", "growth_efficiency", "max_size", "max_vitality",
                "aging_rate", "health_modifier", "max_energy_reserves", "lifespan",
                "cold_resistance", "heat_resistance", "optimal_temperature"
            ]

        original_values = {attr: getattr(genes, attr) for attr in attr_names if hasattr(genes, attr)}

        for i, attr in enumerate(attr_names):
            if i >= len(individual) or not hasattr(genes, attr):
                continue

            if random.random() < indpb:
                original_value = original_values[attr]
                min_val, max_val = gene_ranges[attr]
                range_size = max_val - min_val

                change_limit = max(self.min_absolute_change, original_value * self.max_change_percent)

                if range_size > 10 and change_limit > range_size * 0.1:
                    change_limit = range_size * 0.05

                change = random.uniform(-change_limit, change_limit)
                new_value = original_value + change

                new_value = max(min_val, min(new_value, max_val))
                setattr(genes, attr, new_value)

                self._logger.debug(
                    f"Mutación en '{attr}': original={original_values[attr]:.3f}, "
                    f"average_lifespan={self.avg_lifespan:.3f}, range=({min_val}, {max_val}), "
                    f"min_change={self.min_absolute_change:.3f}, max_change={self.max_change_percent:.3f}, "
                    f"change={change:.3f}, new_value={new_value:.3f}"
                )

        if entity_type == EntityType.FLORA:
            mutated_individual = flora_genes_to_individual(genes)
        else:
            mutated_individual = fauna_genes_to_individual(genes)

        return mutated_individual,