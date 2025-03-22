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

from biome.components.environmental.weather_adaptation import WeatherAdaptationComponent
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.photosynthetic_metabolism import PhotosyntheticMetabolismComponent
from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from biome.components.physiological.vital import VitalComponent
from biome.entities.entity import Entity
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes.fauna_genes import FaunaGenes
from biome.systems.evolution.genes.flora_genes import FloraGenes
from biome.systems.evolution.genes.genes import Genes
from shared.enums.enums import ComponentType, EntityType
from shared.evolution.ranges import FLORA_GENE_RANGES, FAUNA_GENE_RANGES
from shared.types import ClimateData, EntityList


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

def extract_genes_from_fauna(fauna_entity: Entity) -> FaunaGenes:
    genes = FaunaGenes()
    extract_common_genes(fauna_entity, genes)
    # TODO: Específicos de fauna, cuando los tenga
    return genes

def extract_genes_from_flora(flora_entity: Entity) -> FloraGenes:
    genes = FloraGenes()
    extract_common_genes(flora_entity, genes)

    metabolic_component: PhotosyntheticMetabolismComponent = flora_entity.get_component(ComponentType.PHOTOSYNTHETIC_METABOLISM)
    if metabolic_component:
        genes.base_photosynthesis_efficiency = metabolic_component.base_photosynthesis_efficiency
        genes.base_respiration_rate = metabolic_component.base_respiration_rate
        genes.metabolic_activity = metabolic_component.metabolic_activity
        genes.max_energy_reserves = metabolic_component.max_energy_reserves

    nutritional_component: AutotrophicNutritionComponent = flora_entity.get_component(ComponentType.AUTOTROPHIC_NUTRITION)
    if nutritional_component:
        genes.nutrient_absorption_rate = nutritional_component.nutrient_absorption_rate
        genes.mycorrhizal_rate = nutritional_component.mycorrhizal_rate
        genes.base_nutritive_value = nutritional_component.base_nutritive_value
        genes.base_toxicity = nutritional_component.base_toxicity

    return genes

def extract_genes_from_entity(entity: Entity) -> Genes:
    if entity.get_type() == EntityType.FLORA:
        return extract_genes_from_flora(entity)
    elif entity.get_type() == EntityType.FAUNA:
        return extract_genes_from_fauna(entity)
    else:
        raise ValueError(f"Unsupported entity type: {entity.get_type()}")


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

        fauna_genes.cold_resistance,  # 0-1
        fauna_genes.heat_resistance,  # 0-1

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

    genes.cold_resistance = clamped_individual[8]  # 0-1
    genes.heat_resistance = clamped_individual[9]  # 0-1

    genes.optimal_temperature = -20.0 + (clamped_individual[10] * 60.0)  # -20 a 40

    # TODO: Especificos de fauna cuando  haga
    # genes.foraging_efficiency = 0.1 + (clamped_individual[12] * 0.9)  # 0.1-1.0
    # genes.predator_avoidance = clamped_individual[13]  # 0.0-1.0

    return genes

def flora_genes_to_individual(flora_genes: FloraGenes):

    flora_genes.validate_genes()

    return creator.Individual([
        (flora_genes.growth_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (flora_genes.growth_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (flora_genes.max_size - 0.1) / 4.9,  # De 0.1-5.0 a 0-1
        (flora_genes.max_vitality - 50.0) / 150.0,  # De 50-200 a 0-1
        (flora_genes.aging_rate - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (flora_genes.health_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
        (flora_genes.base_photosynthesis_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (flora_genes.base_respiration_rate - 0.01) / 0.99,  # De 0.01-1.0 a 0-1
        (flora_genes.lifespan - 1.0) / 999.0,  # De 1-1000 a 0-1
        (flora_genes.metabolic_activity - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (flora_genes.max_energy_reserves - 50.0) / 100.0,  # De 50-150 a 0-1

        flora_genes.cold_resistance,
        flora_genes.heat_resistance,

        (flora_genes.optimal_temperature + 30.0) / 80.0,  # De -30-50 a [0,1]

        (flora_genes.nutrient_absorption_rate - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (flora_genes.mycorrhizal_rate - 0.01) / 0.03,  # De 0.01-0.04 a 0-1
        (flora_genes.base_nutritive_value - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
        (flora_genes.base_toxicity - 0.01) / 0.99  # De 0.01-1.0 a 0-1
    ])


def deap_genes_to_flora_genes(individual) -> FloraGenes:
    genes = FloraGenes()

    clamped_individual = [max(0.0, min(1.0, val)) for val in individual]

    genes.growth_modifier = 0.1 + (clamped_individual[0] * 1.9)  # 0.1-2.0
    genes.growth_efficiency = 0.1 + (clamped_individual[1] * 0.9)  # 0.1-1.0
    genes.max_size = 0.1 + (clamped_individual[2] * 4.9)  # 0.1-5.0
    genes.max_vitality = 50.0 + (clamped_individual[3] * 150.0)  # 50-200
    genes.aging_rate = 0.1 + (clamped_individual[4] * 1.9)  # 0.1-2.0
    genes.health_modifier = 0.1 + (clamped_individual[5] * 1.9)  # 0.1-2.0
    genes.base_photosynthesis_efficiency = 0.1 + (clamped_individual[6] * 0.9)  # 0.1-1.0
    genes.base_respiration_rate = 0.01 + (clamped_individual[7] * 0.99)  # 0.01-1.0
    genes.lifespan = 1.0 + (clamped_individual[8] * 999.0)  # 1-1000 años
    genes.metabolic_activity = 0.1 + (clamped_individual[9] * 0.9)  # 0.1-1.0
    genes.max_energy_reserves = 50.0 + (clamped_individual[10] * 100.0)  # 50-150

    genes.cold_resistance = clamped_individual[11]  #  0-1
    genes.heat_resistance = clamped_individual[12]  #  0-1

    genes.optimal_temperature = -30 + (clamped_individual[13] * 80.0)  # -30 a 50

    genes.nutrient_absorption_rate = 0.1 + (clamped_individual[14] * 0.9)  # 0.1-1.0
    genes.mycorrhizal_rate = 0.01 + (clamped_individual[15] * 0.03)  # 0.01-0.04
    genes.base_nutritive_value = 0.1 + (clamped_individual[16] * 0.9)  # 0.1-1.0
    genes.base_toxicity = 0.01 + (clamped_individual[17] * 0.99)  # 0.01-1.0

    return genes


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

        def mutation_controlled(individual, indpb, entity_type=None, max_change_percent=0.20, min_absolute_change=0.02):
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

            adaptive_boost_attrs = ["cold_resistance", "heat_resistance"]

            for i, attr in enumerate(attr_names):
                if i >= len(individual) or not hasattr(genes, attr):
                    continue

                if random.random() < indpb:
                    original_value = original_values[attr]
                    min_val, max_val = gene_ranges[attr]
                    range_size = max_val - min_val

                    apply_boost = False
                    if attr in adaptive_boost_attrs:
                        if 0 < original_value < (min_val + range_size * 0.1):
                            apply_boost = False

                    if apply_boost:
                        if random.random() < 0.8:  # Mayor probabilidad de incremento
                            boost_change = random.uniform(0.05 * range_size, 0.4 * range_size)
                            new_value = original_value + boost_change
                        else:
                            new_value = max(min_val, original_value * 0.9)
                    else:
                        change_limit = max(min_absolute_change, original_value * max_change_percent)

                        if range_size > 10 and change_limit > range_size * 0.1:
                            change_limit = range_size * 0.1

                        change = random.uniform(-change_limit, change_limit)
                        new_value = original_value + change

                    new_value = max(min_val, min(new_value, max_val))
                    setattr(genes, attr, new_value)

            if entity_type == EntityType.FLORA:
                mutated_individual = flora_genes_to_individual(genes)
            else:
                mutated_individual = fauna_genes_to_individual(genes)

            return mutated_individual,

        self.toolbox.register("mutate", mutation_controlled, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evolve_population(self, entities: EntityList, climate_data: ClimateData, generation_count=20, k_best=5):
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


        def eval_individual(individual):
            if entity_type == EntityType.FLORA:
                genes = deap_genes_to_flora_genes(individual)
                return (compute_fitness(genes, climate_data),)
            else:  # FAUNA
                genes = deap_genes_to_fauna_genes(individual)
                return (compute_fitness(genes, climate_data),)

        self.toolbox.register("evaluate", eval_individual)

        def mutate_with_type(individual, indpb):
            return self.toolbox.mutate(individual, indpb, entity_type)

        self.toolbox.register("mutate_typed", mutate_with_type, indpb=0.2)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, self.toolbox,
                            cxpb=0.5, mutpb=0.2,
                            ngen=generation_count,
                            stats=stats, verbose=True)

        top_individuals = tools.selBest(population, k=k_best)

        if entity_type == EntityType.FLORA:
            evolved_genes = [deap_genes_to_flora_genes(ind) for ind in top_individuals]
        else:
            evolved_genes = [deap_genes_to_fauna_genes(ind) for ind in top_individuals]

        self.stats_history.append(stats)

        return evolved_genes