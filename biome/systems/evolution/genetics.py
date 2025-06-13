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
Genetic algorithm implementation for entity evolution using DEAP.

Manages crossover, mutation and selection operations for entities;
handles gene extraction and conversion between entities and chromosomes.
Evolves populations based on fitness evaluation and climate adaptation -
supports adaptive mutation and tracks genetic lineage through generations.
"""

import random

import numpy as np
from deap import algorithms, base, creator, tools

from biome.entities.entity import Entity
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes.fauna_genes import extract_genes_from_fauna
from biome.systems.evolution.genes.flora_genes import extract_genes_from_flora
from biome.systems.evolution.genes.genes import Genes
from biome.systems.evolution.genetic_converter import GeneticConverter
from biome.systems.evolution.mutations.adaptive_mutation_operator import (
    AdaptiveMutationOperator,
)
from biome.systems.evolution.visualization.evo_crossover_tracker import (
    GeneticCrossoverTracker,
)
from shared.enums.enums import EntityType, MutationType
from shared.evolution.ranges import FAUNA_GENE_RANGES, FLORA_GENE_RANGES
from shared.types import ClimateData, EntityList


def extract_genes_from_entity(entity: Entity) -> Genes:
    if entity.get_type() == EntityType.FLORA:
        return extract_genes_from_flora(entity)
    elif entity.get_type() == EntityType.FAUNA:
        return extract_genes_from_fauna(entity)
    else:
        raise ValueError(f"Unsupported entity type: {entity.get_type()}")


class GeneticAlgorithmModel:
    _types_created = False

    def __init__(self, crossover_tracker: GeneticCrossoverTracker = None):
        if not GeneticAlgorithmModel._types_created:
            self._setup_deap()
            GeneticAlgorithmModel._types_created = True

        self.toolbox = base.Toolbox()
        self._setup_toolbox()
        self.stats_history = []
        self._genetic_tracker: GeneticCrossoverTracker = crossover_tracker

    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    def _setup_toolbox(self):
        self.toolbox.register("attr_float", random.random)

        self.toolbox.register("individual_flora", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=len(FLORA_GENE_RANGES))
        self.toolbox.register("individual_fauna", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=len(FAUNA_GENE_RANGES))

        self.toolbox.register("population_flora", tools.initRepeat, list, self.toolbox.individual_flora)
        self.toolbox.register("population_fauna", tools.initRepeat, list, self.toolbox.individual_fauna)

        self.toolbox.register("mate", tools.cxBlend, alpha=0.4)

        # Para que sea adaptativo, al final registro el operador mutate en cada evolve.
        self.toolbox.register("select", tools.selTournament,
                              tournsize=2)  # lo he cambiado de 3 a 2 para ver si la convergencia es más lenta

    def evolve_population(self, entities: EntityList, climate_data: ClimateData, current_evo_cycle: int,
                          generation_count=1, k_best=5):
        if not entities:
            return []

        entity_type = entities[0].get_type()
        entity_species = entities[0].get_species()
        population = []
        for entity in entities:
            current_genes = extract_genes_from_entity(entity)
            population.append(GeneticConverter.genes_to_individual(current_genes, entity_type))

        average_lifespan = np.average(np.array([e.lifespan for e in entities]))
        mutation_adaptive_operator = AdaptiveMutationOperator(indpb=0.15)
        mutation_adaptive_operator.adapt(entity_type, average_lifespan, MutationType.GAUSSIAN)

        self.toolbox.register("mutate", mutation_adaptive_operator)

        def eval_individual(individual):
            genes = GeneticConverter.individual_to_genes(individual, entity_type)
            return (compute_fitness(genes, climate_data),)

        self.toolbox.register("evaluate", eval_individual)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hof = tools.HallOfFame(k_best)
        algorithms.eaSimple(population, self.toolbox,
                            cxpb=0.65, mutpb=0.3,
                            ngen=generation_count,
                            stats=stats, verbose=False, halloffame=hof)

        top_individuals = list(hof)

        if self._genetic_tracker:
            original_genes = []
            for entity in entities[:2]:
                genes = extract_genes_from_entity(entity)
                original_genes.append(genes)

            for i, individual in enumerate(top_individuals):
                evolved_gene = GeneticConverter.individual_to_genes(individual, entity_type)

                self._genetic_tracker.register_crossover(
                    str(entity_species),
                    current_evo_cycle,
                    original_genes,
                    evolved_gene
                )

        evolved_genes = [GeneticConverter.individual_to_genes(ind, entity_type) for ind in top_individuals]

        return evolved_genes
