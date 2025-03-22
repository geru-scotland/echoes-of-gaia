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
from biome.components.physiological.metabolic import MetabolicComponent
from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from biome.components.physiological.vital import VitalComponent
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes import FloraGenes
from biome.entities.flora import Flora
from shared.enums.enums import ComponentType
from shared.evolution.ranges import FLORA_GENE_RANGES


def extract_genes_from_entity(flora_entity: Flora) -> FloraGenes:
    genes = FloraGenes()

    genes.lifespan = flora_entity.lifespan

    growth_component: GrowthComponent = flora_entity.get_component(ComponentType.GROWTH)
    if growth_component:
        genes.growth_modifier = growth_component.growth_modifier
        genes.growth_efficiency = growth_component.growth_efficiency
        genes.max_size = growth_component.max_size

    vital_component: VitalComponent = flora_entity.get_component(ComponentType.VITAL)
    if vital_component:
        genes.max_vitality = vital_component.max_vitality
        genes.aging_rate = vital_component.aging_rate
        genes.health_modifier = vital_component.health_modifier

    metabolic_component: MetabolicComponent = flora_entity.get_component(ComponentType.METABOLIC)
    if metabolic_component:
        genes.base_photosynthesis_efficiency = metabolic_component.base_photosynthesis_efficiency
        genes.base_respiration_rate = metabolic_component.base_respiration_rate
        genes.metabolic_activity = metabolic_component.metabolic_activity
        genes.max_energy_reserves = metabolic_component.max_energy_reserves

    weather_adaptation_component: WeatherAdaptationComponent = flora_entity.get_component(ComponentType.WEATHER_ADAPTATION)
    if weather_adaptation_component:
        genes.cold_resistance = weather_adaptation_component.cold_resistance
        genes.heat_resistance = weather_adaptation_component.heat_resistance
        genes.optimal_temperature = weather_adaptation_component.optimal_temperature

    nutritional_component: AutotrophicNutritionComponent = flora_entity.get_component(ComponentType.AUTOTROPHIC_NUTRITION)
    if nutritional_component:
        genes.nutrient_absorption_rate = nutritional_component.nutrient_absorption_rate
        genes.mycorrhizal_rate = nutritional_component.mycorrhizal_rate
        genes.base_nutritive_value = nutritional_component.base_nutritive_value
        genes.base_toxicity = nutritional_component.base_toxicity

    return genes


def flora_genes_to_individual(flora_genes):

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
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=len(FLORA_GENE_RANGES))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)

        def mutation_controlled(individual, indpb, max_change_percent=0.20, min_absolute_change=0.02):
            flora_genes = deap_genes_to_flora_genes(individual)

            original_values = {
                "growth_modifier": flora_genes.growth_modifier,
                "growth_efficiency": flora_genes.growth_efficiency,
                "max_size": flora_genes.max_size,
                "max_vitality": flora_genes.max_vitality,
                "aging_rate": flora_genes.aging_rate,
                "health_modifier": flora_genes.health_modifier,
                "base_photosynthesis_efficiency": flora_genes.base_photosynthesis_efficiency,
                "base_respiration_rate": flora_genes.base_respiration_rate,
                "lifespan": flora_genes.lifespan,
                "metabolic_activity": flora_genes.metabolic_activity,
                "max_energy_reserves": flora_genes.max_energy_reserves,
                "cold_resistance": flora_genes.cold_resistance,
                "heat_resistance": flora_genes.heat_resistance,
                "optimal_temperature": flora_genes.optimal_temperature,
                "nutrient_absorption_rate": flora_genes.nutrient_absorption_rate,
                "mycorrhizal_rate": flora_genes.mycorrhizal_rate,
                "base_nutritional_value": flora_genes.base_nutritive_value,
                "base_toxicity": flora_genes.base_toxicity
            }

            adaptive_boost_attrs = ["cold_resistance", "heat_resistance"]

            for i, attr in enumerate([
                "growth_modifier", "growth_efficiency", "max_size", "max_vitality",
                "aging_rate", "health_modifier", "base_photosynthesis_efficiency",
                "base_respiration_rate", "lifespan", "metabolic_activity",
                "max_energy_reserves", "cold_resistance", "heat_resistance",
                "optimal_temperature", "nutrient_absorption_rate", "mycorrhizal_rate",
                "base_nutritional_value", "base_toxicity"
            ]):
                if random.random() < indpb:
                    original_value = original_values[attr]
                    min_val, max_val = FLORA_GENE_RANGES[attr]
                    range_size = max_val - min_val

                    apply_boost = False
                    if attr in adaptive_boost_attrs:
                        # Si el valor es muy bajo (menos del 10%) y positivo
                        # aplico el impulso adaptativo
                        if 0 < original_value < (min_val + range_size * 0.1):
                            apply_boost = True

                    if apply_boost:
                        if random.random() < 0.8:  # Más prob de incremento
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

                    setattr(flora_genes, attr, new_value)

            mutated_individual = flora_genes_to_individual(flora_genes)

            return mutated_individual,

        self.toolbox.register("mutate", mutation_controlled, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evolve_population(self, current_flora, climate_data, generation_count=20, k_best=5):
        population = []
        for flora in current_flora:
            current_genes: FloraGenes = extract_genes_from_entity(flora)
            population.append(flora_genes_to_individual(current_genes))

        # if len(population) < 10:
        #     random_individuals = self.toolbox.population(10 - len(population))
        #     population.extend(random_individuals)

        def eval_individual(individual):
            flora_genes = deap_genes_to_flora_genes(individual)
            return (compute_fitness(flora_genes, climate_data),)

        self.toolbox.register("evaluate", eval_individual)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaSimple(population, self.toolbox,
                            cxpb=0.5, mutpb=0.2,
                            ngen=generation_count,
                            stats=stats, verbose=True)

        top_individuals = tools.selBest(population, k=k_best)

        evolved_genes = [deap_genes_to_flora_genes(ind) for ind in top_individuals]

        self.stats_history.append(stats)

        return evolved_genes

