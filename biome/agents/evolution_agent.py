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
import itertools
import random
from logging import Logger
from typing import List, Dict, Any

import numpy as np
from pandas import DataFrame

from biome.entities.fauna import Fauna
from biome.systems.events.event_bus import BiomeEventBus
from shared.enums.events import BiomeEvent
from shared.enums.enums import FloraSpecies, EntityType, FaunaSpecies
from biome.entities.flora import Flora
from biome.agents.base import Agent, TAction, TState
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes.flora_genes import FloraGenes
from biome.systems.evolution.genetics import GeneticAlgorithmModel, extract_genes_from_entity
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.strings import Loggers
from shared.timers import Timers
from shared.types import Observation, EntityList
from utils.loggers import LoggerManager


class EvolutionAgentAI(Agent):
    def __init__(self, climate_data_manager: ClimateDataManager, entity_provider: EntityProvider, entity_type: EntityType,
                 species: FloraSpecies | FaunaSpecies, base_lifespan: float, evolution_cycle_time: int = Timers.Agents.Evolution.EVOLUTION_CYCLE,
                 evolution_registry = None):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self.entity_provider: EntityProvider = entity_provider
        self._genetic_model: GeneticAlgorithmModel = GeneticAlgorithmModel()
        self._evolution_cycle_time = evolution_cycle_time
        self._evolution_cycle: itertools.count[int] = itertools.count(0)
        self._current_evolution_cycle: int = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)
        self._entity_type: EntityType = entity_type
        self._species: FloraSpecies = species
        self._species_base_lifespan: float = base_lifespan
        self._logger.info(f"Initialized Evolution Agent for species: {species}")
        self._evolution_registry = evolution_registry

    def perceive(self) -> Observation:
        if self._entity_type == EntityType.FLORA:
            entities = self.entity_provider.get_flora(only_alive=True)
        else:
            entities = self.entity_provider.get_fauna(only_alive=True)

        species_entities = [entity for entity in entities if entity.get_species() == self._species]

        return {
            "climate_data": self._climate_data_manager.get_data(self._current_evolution_cycle)[-1:],
            "entities": species_entities
        }

    def decide(self, observation: Observation) -> TAction:
        self._logger.info(f"Making evolution decision for {self._entity_type} species: {self._species}")

        climate_data = observation["climate_data"]
        entities = observation["entities"]
        self._logger.debug(f"EVOLUTION AGENT GOT: {len(entities)} alive {self._entity_type} entities")

        entities_by_species = {}
        for entity in entities:
            species = entity.get_species()
            if species not in entities_by_species:
                entities_by_species[species] = []
            entities_by_species[species].append(entity)

        all_evolved_genes = []
        entities_to_remove = []

        for species, entities_list in entities_by_species.items():
            entities_sorted = sorted(entities_list,
                                     key=lambda e: self._calculate_entity_fitness(e, climate_data))

            removal_count = max(1, int(len(entities_list) * 0.2))
            for i in range(removal_count):
                if i < len(entities_sorted):
                    entities_to_remove.append(entities_sorted[i].get_id())

            k_best = self._compute_k_best(entities)

            evolved_genes = self._genetic_model.evolve_population(
                entities_list, climate_data, generation_count=10, k_best=k_best
            )

            for genes in evolved_genes:
                all_evolved_genes.append((species, genes))

        return {
            "evolved_genes": all_evolved_genes,
            "entities_to_remove": entities_to_remove
        }

    def act(self, action: TAction) -> None:
        for entity_id in action["entities_to_remove"]:
            pass

        self._current_evolution_cycle = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)

        for species, genes in action["evolved_genes"]:
            if species == self._species:
                self._create_evolved_entity(species, genes)

        self._evolution_registry.record_generation(self._current_evolution_cycle)

        # Nueva generación spawneada, calculo lifespan medio
        average_lifespan: float = self._compute_current_generation_lifespan()
        self._increase_evolution_time_cycle(average_lifespan)

    def _compute_k_best(self, entities: EntityList) -> int:
        avg_lifespan: float = np.average(np.array([e.lifespan for e in entities]))
        population_size: float = len(entities)
        base_k_best = max(3, min(10, int(population_size * random.uniform(0.15, 0.25))))

        base_lifespan = self._species_base_lifespan
        multiplier = 1 + ((avg_lifespan - base_lifespan) / base_lifespan) * 0.1

        multiplier = max(0.8, min(multiplier, 1.2))

        return int(base_k_best * multiplier)

    def _compute_current_generation_lifespan(self) -> float:
        if self._entity_type == EntityType.FLORA:
            entities = self.entity_provider.get_flora(only_alive=True)
        else:
            entities = self.entity_provider.get_fauna(only_alive=True)

        generation_lifespans = [entity.lifespan for entity in entities
                                if entity.get_species() == self._species]

        if not generation_lifespans:
            return self._species_base_lifespan

        generation_average_lifespan = np.average(np.array(generation_lifespans))
        self._logger.debug(f"CURRENT GENERATION AVERAGE LIFESPAN: {generation_average_lifespan}")

        return float(generation_average_lifespan * Timers.Calendar.YEAR)

    def _calculate_entity_fitness(self, entity, climate_data):
        genes = extract_genes_from_entity(entity)
        return compute_fitness(genes, climate_data)

    def _create_evolved_entity(self, species, genes: FloraGenes | FaunaSpecies) -> None:
        try:
            components: List[Dict[str, Any]] = genes.convert_genes_to_components()
            current_cycle = self._current_evolution_cycle

            self._logger.debug(f"=== EVOLVED ENTITY [{species}] - Generation {current_cycle} ===")
            self._logger.debug(f"Growth: mod={genes.growth_modifier:.2f}, eff={genes.growth_efficiency:.2f}")
            self._logger.debug(f"Size: max={genes.max_size:.2f}")
            self._logger.debug(f"Vitality: max={genes.max_vitality:.2f}, aging={genes.aging_rate:.2f}")

            if hasattr(genes, 'base_photosynthesis_efficiency'):
                self._logger.debug(
                    f"Photosynthesis: eff={genes.base_photosynthesis_efficiency:.2f}, resp={genes.base_respiration_rate:.4f}")

            self._logger.debug(f"Climate: cold_res={genes.cold_resistance:.2f}, heat_res={genes.heat_resistance:.2f}")
            self._logger.debug(f"Optimal temp: {genes.optimal_temperature:.1f}°C")

            if self._entity_type == EntityType.FLORA:
                entity_class = Flora
                entity_species_enum = FloraSpecies
            else:
                entity_class = Fauna
                entity_species_enum = FaunaSpecies

            BiomeEventBus.trigger(
                BiomeEvent.CREATE_ENTITY,
                entity_class=entity_class,
                entity_species_enum=entity_species_enum,
                species_name=str(species),
                lifespan=genes.lifespan,
                custom_components=components,
                evolution_cycle=current_cycle,
            )

        except Exception as e:
            self._logger.exception(f"Error al crear entidad evolucionada de especie {species}: {e}")

    def get_species(self) -> FloraSpecies:
        return self._species

    def get_evolution_cycle_time(self) -> int:
        return self._evolution_cycle_time

    def _increase_evolution_time_cycle(self, average_lifespan: float = None) -> None:
        lifespan: float = average_lifespan if average_lifespan else self._species_base_lifespan
        # que sea dinámico, si el lifespan anterior es más que el actual, que lo vuelva a hacer
        if self._evolution_cycle_time < 0.4 * lifespan:
            self._evolution_cycle_time += lifespan * random.uniform(0.01, 0.03)

    @property
    def entity_type(self) -> EntityType:
        return self._entity_type