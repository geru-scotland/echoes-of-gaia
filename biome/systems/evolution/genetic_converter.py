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
Utility class for genetic algorithm data format conversions.

Handles conversion between gene objects and DEAP individuals;
normalizes gene values based on predefined attribute ranges.
Supports flora and fauna genetic representations with automatic validation.
"""


from deap import creator

from biome.systems.evolution.genes.fauna_genes import FaunaGenes
from biome.systems.evolution.genes.flora_genes import FloraGenes
from shared.enums.enums import EntityType
from shared.evolution.ranges import FAUNA_GENE_RANGES, FLORA_GENE_RANGES


class GeneticConverter:
    @staticmethod
    def genes_to_individual(genes, entity_type: EntityType):
        gene_ranges = FLORA_GENE_RANGES if entity_type == EntityType.FLORA else FAUNA_GENE_RANGES
        normalized_genes = []

        for attr_name, (min_val, max_val) in gene_ranges.items():
            if hasattr(genes, attr_name):
                value = getattr(genes, attr_name)

                value = max(min_val, min(value, max_val))

                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                normalized_genes.append(normalized)

        return creator.Individual(normalized_genes)

    @staticmethod
    def individual_to_genes(individual, entity_type: EntityType):
        genes_class = FloraGenes if entity_type == EntityType.FLORA else FaunaGenes
        gene_ranges = FLORA_GENE_RANGES if entity_type == EntityType.FLORA else FAUNA_GENE_RANGES

        genes = genes_class()
        attr_names = list(gene_ranges.keys())

        for i, attr_name in enumerate(attr_names):
            if i < len(individual):
                min_val, max_val = gene_ranges[attr_name]

                norm_value = max(0.0, min(1.0, individual[i]))

                actual_value = min_val + norm_value * (max_val - min_val)
                setattr(genes, attr_name, actual_value)

        genes.validate_genes()
        return genes