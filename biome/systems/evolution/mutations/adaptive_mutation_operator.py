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
import math
import random
from logging import Logger

from deap import tools

from biome.systems.evolution.genetic_converter import GeneticConverter
from shared.enums.enums import EntityType
from shared.enums.strings import Loggers
from shared.evolution.ranges import FLORA_GENE_RANGES, FAUNA_GENE_RANGES


import random

from utils.loggers import LoggerManager

from enum import Enum, auto


class MutationType(Enum):
    ADAPTIVE = auto()
    GAUSSIAN = auto()


class AdaptiveMutationOperator:
    def __init__(self, indpb: float = 0.1):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.indpb = indpb
        self.entity_type = None
        self.avg_lifespan = 10.0
        self.max_change_percent = 0.1
        self.min_absolute_change = 0.01
        self.mutation_type = MutationType.ADAPTIVE
        self.sigma_factor = 0.05

    def adapt(self, entity_type: EntityType, avg_lifespan: float,
              mutation_type: MutationType = MutationType.ADAPTIVE) -> None:
        self.entity_type = entity_type
        self.avg_lifespan = avg_lifespan
        self.mutation_type = mutation_type

        max_percent = 0.2
        min_percent = 0.01
        # Pongo 40, pero es un poco por los datos actuales para pruebas, cambiar esto.
        scale = 40.0

        # Cuanto mayor es avg_lifespan, más cerca del maxpercent; cuanto menor, más cerca del 1%
        ratio = min(1.0, avg_lifespan / scale)
        self.max_change_percent = max_percent

        self.sigma_factor = 0.03 + (ratio * 0.05)  # Entre 0.03 y 0.08

    def __call__(self, individual, indpb=None):
        indpb = indpb if indpb is not None else self.indpb

        if self.entity_type is None:
            self.entity_type = (EntityType.FLORA
                                if len(individual) == len(FLORA_GENE_RANGES)
                                else EntityType.FAUNA)

        if self.mutation_type == MutationType.GAUSSIAN:
            return self._gaussian_mutation(individual, indpb)
        else:
            return self._adaptive_mutation(individual, indpb)

    def _adaptive_mutation(self, individual, indpb):
        genes = GeneticConverter.individual_to_genes(individual, self.entity_type)

        gene_ranges = FLORA_GENE_RANGES if self.entity_type == EntityType.FLORA else FAUNA_GENE_RANGES
        attr_names = list(gene_ranges.keys())

        self._logger.debug(f"===== ADAPTATIVE MUTATION STARTS =====")
        self._logger.debug(f"Entity Type: {self.entity_type}, Avg Lifespan: {self.avg_lifespan:.2f}")
        self._logger.debug(
            f"Max Change %: {self.max_change_percent:.4f}, Min Abs Change: {self.min_absolute_change:.4f}")

        original_values = {attr: getattr(genes, attr) for attr in attr_names if hasattr(genes, attr)}

        for attr_name in attr_names:
            if not hasattr(genes, attr_name) or random.random() >= indpb:
                continue

            original_value = getattr(genes, attr_name)
            min_val, max_val = gene_ranges[attr_name]
            range_size = max_val - min_val

            change_limit = max(self.min_absolute_change, original_value * self.max_change_percent)

            if range_size > 10 and change_limit > range_size * 0.1:
                change_limit = range_size * 0.05

            change = random.uniform(-change_limit, change_limit)
            new_value = original_value + change

            new_value = max(min_val, min(new_value, max_val))

            setattr(genes, attr_name, new_value)

            self._logger.debug(
                f"Mutation'{attr_name}': orig={original_value:.3f}, "
                f"change={change:.3f} (±{change_limit:.3f}), "
                f"new={new_value:.3f}, range=[{min_val}, {max_val}]"
            )

        mutated_individual = GeneticConverter.genes_to_individual(genes, self.entity_type)

        self._logger.debug(f"===== END ADAPTIVE MUTATIONN =====")

        return mutated_individual,

    def _gaussian_mutation(self, individual, indpb):
        genes = GeneticConverter.individual_to_genes(individual, self.entity_type)

        gene_ranges = FLORA_GENE_RANGES if self.entity_type == EntityType.FLORA else FAUNA_GENE_RANGES
        attr_names = list(gene_ranges.keys())

        self._logger.debug(f"===== GAUSSIAN MUTATIN STARTS =====")
        self._logger.debug(f"Entity Type: {self.entity_type}, Sigma Factor: {self.sigma_factor:.4f}")

        original_values = {attr: getattr(genes, attr) for attr in attr_names if hasattr(genes, attr)}

        for attr_name in attr_names:
            if not hasattr(genes, attr_name) or random.random() >= indpb:
                continue

            original_val = getattr(genes, attr_name)
            min_val, max_val = gene_ranges[attr_name]
            range_size = max_val - min_val

            sigma = self.sigma_factor * range_size

            mutation = random.gauss(0, sigma)
            mutated_val = original_val + mutation

            mutated_val = max(min_val, min(mutated_val, max_val))
            setattr(genes, attr_name, mutated_val)

            self._logger.debug(
                f"Mutation'{attr_name}': orig={original_val:.3f}, "
                f"change={mutation:.3f} (σ={sigma:.3f}), "
                f"new={mutated_val:.3f}, range=[{min_val}, {max_val}]"
            )

        mutated_individual = GeneticConverter.genes_to_individual(genes, self.entity_type)

        self._logger.debug(f"===== END GAUSSIAN MUTATION =====")

        return mutated_individual,