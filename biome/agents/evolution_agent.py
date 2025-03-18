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
from logging import Logger
from typing import List, Dict, Any

from pandas import DataFrame

from biome.systems.events.event_bus import BiomeEventBus
from shared.enums.events import BiomeEvent
from shared.enums.enums import FloraSpecies
from biome.entities.flora import Flora
from biome.agents.base import Agent, TAction, TState
from biome.systems.evolution.fitness import compute_fitness
from biome.systems.evolution.genes import FloraGenes
from biome.systems.evolution.genetics import GeneticAlgorithmModel
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.strings import Loggers
from shared.types import Observation, EntityList
from utils.loggers import LoggerManager


class EvolutionAgentAI(Agent):
    def __init__(self, climate_data_manager: ClimateDataManager, entity_provider: EntityProvider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self.entity_provider: EntityProvider = entity_provider
        self._genetic_model: GeneticAlgorithmModel = GeneticAlgorithmModel()
        self._evolution_cycle: itertools.count[int] = itertools.count(0)
        self._current_evolution_cycle: int = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)

    def perceive(self) -> Observation:
        return {
            "climate_data": self._climate_data_manager.get_data(self._current_evolution_cycle)[-1:],
            "flora": self.entity_provider.get_flora()
        }

    def decide(self, observation: Observation) -> TAction:
       self._logger.info(f"Observation: {observation}")
       climate_data = observation["climate_data"]
       flora_entities = observation["flora"]

       flora_by_species = {}
       for entity in flora_entities:
           species = entity.get_species()
           if species not in flora_by_species:
               flora_by_species[species] = []
           flora_by_species[species].append(entity)

       all_evolved_genes = []
       entities_to_remove = []

       for species, entities in flora_by_species.items():
           entities_sorted = sorted(entities,
                                    key=lambda e: self._calculate_entity_fitness(e, climate_data))

           removal_count = max(1, int(len(entities) * 0.2))
           for i in range(removal_count):
               if i < len(entities_sorted):
                   entities_to_remove.append(entities_sorted[i].get_id())

           evolved_genes = self._genetic_model.evolve_population(
               entities, climate_data, generation_count=5
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

        for species, genes in action["evolved_genes"]:
            self._create_evolved_entity(species, genes)

        self._current_evolution_cycle = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)

    def _calculate_entity_fitness(self, entity, climate_data):
        genes = self._genetic_model.extract_genes_from_entity(entity)

        return compute_fitness(genes, climate_data)

    def _create_evolved_entity(self, species, genes: FloraGenes) -> None:
        try:
            components = self._convert_genes_to_components(genes)

            BiomeEventBus.trigger(
                BiomeEvent.CREATE_ENTITY,
                entity_class=Flora,
                entity_species_enum=FloraSpecies,
                species_name=str(species),
                lifespan=genes.lifespan,
                custom_components=components,
            )

        except Exception as e:
            self._logger.exception(f"Error al crear entidad evolucionada de especie {species}: {e}")

    def _convert_genes_to_components(self, genes: FloraGenes):
        components = []

        growth_component: Dict[str, Any] = {
            "GrowthComponent": {
                "growth_modifier": genes.growth_modifier,
                "growth_efficiency": genes.growth_efficiency,
                "lifespan": genes.lifespan,
                "max_size": genes.max_size * 5.0
            }
        }
        components.append(growth_component)

        vital_component: Dict[str, Any] = {
            "VitalComponent": {
                "max_vitality": genes.max_vitality,
                "aging_rate": genes.aging_rate,
                "lifespan": genes.lifespan,
                "health_modifier": genes.health_modifier,
            }
        }
        components.append(vital_component)

        metabolic_component: Dict[str, Any] = {
            "MetabolicComponent": {
                "photosynthesis_efficiency": genes.base_photosynthesis_efficiency,
                "respiration_rate": genes.base_respiration_rate,
                "metabolic_activity": genes.metabolic_activity,
                "max_energy_reserves": genes.max_energy_reserves,
            }
        }
        components.append(metabolic_component)

        weather_adaptation_component: Dict[str, Any] = {
            "WeatherAdaptationComponent": {
                "cold_resistance": genes.cold_resistance,
                "heat_resistance": genes.heat_resistance,
                "optimal_temperature": genes.optimal_temperature,
            }
        }
        components.append(weather_adaptation_component)

        return components