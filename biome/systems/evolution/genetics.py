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

import numpy as np
from deap import base, creator, tools, algorithms

from biome.entities.entity import Entity
from biome.systems.evolution.adaptive_mutation_operator import AdaptiveMutationOperator
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes.fauna_genes import fauna_genes_to_individual, deap_genes_to_fauna_genes, \
    extract_genes_from_fauna
from biome.systems.evolution.genes.flora_genes import flora_genes_to_individual, deap_genes_to_flora_genes, \
    extract_genes_from_flora
from biome.systems.evolution.genes.genes import Genes
from shared.enums.enums import EntityType
from shared.evolution.ranges import FLORA_GENE_RANGES, FAUNA_GENE_RANGES
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

    def __init__(self):
        if not GeneticAlgorithmModel._types_created:
            self._setup_deap()
            GeneticAlgorithmModel._types_created = True

        self.toolbox = base.Toolbox()
        self._setup_toolbox()
        self.stats_history = []

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

        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)

        # Para que sea adaptativo, al final registro el operador mutate en cada evolve.
        self.toolbox.register("select", tools.selTournament, tournsize=2) # lo he cambiado de 3 a 2 para ver si la convergencia es más lenta

    def evolve_population(self, entities: EntityList, climate_data: ClimateData, generation_count=1, k_best=5):
        if not entities:
            return []

        entity_type = entities[0].get_type()

        population = []
        for entity in entities:
            current_genes = extract_genes_from_entity(entity)
            if entity_type == EntityType.FLORA:
                population.append(flora_genes_to_individual(current_genes))
            else:
                population.append(fauna_genes_to_individual(current_genes))

        average_lifespan: float = np.average(np.array([e.lifespan for e in entities]))
        mutation_adaptive_operator: AdaptiveMutationOperator = AdaptiveMutationOperator(indpb=0.01)
        mutation_adaptive_operator.adapt(entity_type, average_lifespan)
        self.toolbox.register("mutate", mutation_adaptive_operator)

        def eval_individual(individual):
            if entity_type == EntityType.FLORA:
                genes = deap_genes_to_flora_genes(individual)
            else:  # FAUNA
                genes = deap_genes_to_fauna_genes(individual)

            return (compute_fitness(genes, climate_data),)

        self.toolbox.register("evaluate", eval_individual)

        stats = tools.Statistics(lambda ind: ind.fitness.values)

        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, self.toolbox,
                            cxpb=0.4, mutpb=0.2,
                            ngen=generation_count,
                            stats=stats, verbose=False)

        top_individuals = tools.selBest(population, k=k_best)

        if entity_type == EntityType.FLORA:
            evolved_genes = [deap_genes_to_flora_genes(ind) for ind in top_individuals]
        else:
            evolved_genes = [deap_genes_to_fauna_genes(ind) for ind in top_individuals]

        self.stats_history.append(stats)

        return evolved_genes