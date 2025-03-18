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
from biome.components.physiological.vital import VitalComponent
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes import FloraGenes
from biome.entities.flora import Flora
from shared.enums.enums import ComponentType


class GeneticAlgorithmModel:
    def __init__(self):
        self._setup_deap()

        self.stats_history = []

    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_float", random.random)

        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=14)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _validate_genes(self, genes: FloraGenes) -> FloraGenes:
        valid_ranges = {
            "growth_modifier": (0.1, 2.0),
            "growth_efficiency": (0.3, 1.0),
            "max_size": (0.1, 5.0),
            "max_vitality": (50.0, 200.0),
            "aging_rate": (0.1, 2.0),
            "health_modifier": (0.4, 2.0),
            "base_photosynthesis_efficiency": (0.4, 1.0),
            "base_respiration_rate": (0.05, 1.0),
            "lifespan": (1.0, 1000.0),  # en años
            "metabolic_activity": (0.2, 1.0),
            "max_energy_reserves": (50.0, 150.0),
            "cold_resistance": (0.0, 1.0),
            "heat_resistance": (0.0, 1.0),
            "optimal_temperature": (-30, 50)
        }

        for attr, (min_val, max_val) in valid_ranges.items():
            if hasattr(genes, attr):
                current_val = getattr(genes, attr)
                valid_val = max(min_val, min(current_val, max_val))
                setattr(genes, attr, valid_val)

        return genes

    def evolve_population(self, current_flora, climate_data, generation_count=20):
        population = []
        for flora in current_flora:
            current_genes: FloraGenes = self.extract_genes_from_entity(flora)
            population.append(self._flora_genes_to_individual(current_genes))

        # if len(population) < 10:
        #     random_individuals = self.toolbox.population(10 - len(population))
        #     population.extend(random_individuals)

        def eval_individual(individual):
            flora_genes = self._deap_genes_to_flora_genes(individual)
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

        top_individuals = tools.selBest(population, k=5)

        evolved_genes = [self._deap_genes_to_flora_genes(ind) for ind in top_individuals]

        self.stats_history.append(stats)

        return evolved_genes

    def _deap_genes_to_flora_genes(self, individual) -> FloraGenes:
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

        return genes

    def _flora_genes_to_individual(self, flora_genes):
        validated_genes = self._validate_genes(flora_genes)

        return creator.Individual([
            (validated_genes.growth_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
            (validated_genes.growth_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
            (validated_genes.max_size - 0.1) / 4.9,  # De 0.1-5.0 a 0-1
            (validated_genes.max_vitality - 50.0) / 150.0,  # De 50-200 a 0-1
            (validated_genes.aging_rate - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
            (validated_genes.health_modifier - 0.1) / 1.9,  # De 0.1-2.0 a 0-1
            (validated_genes.base_photosynthesis_efficiency - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
            (validated_genes.base_respiration_rate - 0.01) / 0.99,  # De 0.01-1.0 a 0-1
            (validated_genes.lifespan - 1.0) / 999.0,  # De 1-1000 a 0-1
            (validated_genes.metabolic_activity - 0.1) / 0.9,  # De 0.1-1.0 a 0-1
            (validated_genes.max_energy_reserves - 50.0) / 100.0,  # De 50-150 a 0-1

            validated_genes.cold_resistance,
            validated_genes.heat_resistance,

            (validated_genes.optimal_temperature + 30.0) / 80.0  # De -30-50 a [0,1]

        ])

    def extract_genes_from_entity(self, flora_entity: Flora):
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

        return genes