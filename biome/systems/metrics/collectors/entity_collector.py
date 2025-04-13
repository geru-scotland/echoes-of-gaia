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
from collections import defaultdict
from logging import Logger
from typing import Dict, Any, List

import numpy as np
from dotenv.cli import enumerate_env

from biome.systems.evolution.registry import EvolutionAgentRegistry
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.enums import ComponentType, DietType
from shared.enums.strings import Loggers
from shared.types import EntityList
from utils.loggers import LoggerManager


class EntityDataCollector:

    def __init__(self, entity_provider: EntityProvider, climate_manager: ClimateDataManager,
                 evolution_registry: EvolutionAgentRegistry):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._entity_provider: EntityProvider = entity_provider
        self._evolution_registry: EvolutionAgentRegistry = evolution_registry
        self._climate_data_manager: ClimateDataManager = climate_manager

    def collect_data(self) -> Dict[str, int | float]:
        self._logger.debug("EntityDataCollectar has started collecting data...")
        flora, fauna = self._entity_provider.get_entities(only_alive=True)

        if not flora and not fauna:
            return self._get_empty_stats()

        all_entities: EntityList = flora + fauna
        all_attributes: Dict[str, List[Any]] = self._collect_all_attributes(all_entities)
        stats: Dict[str, int | float] = self._compute_statistics(all_attributes)
        self._logger.debug(f"Final stats: {stats}")

        herbivore_count = 0
        carnivore_count = 0
        omnivore_count = 0

        flora_by_species = {}
        fauna_by_species = {}

        for entity in flora:
            species = str(entity.get_species())
            flora_by_species[species] = flora_by_species.get(species, 0) + 1

        for entity in fauna:
            species = str(entity.get_species())
            fauna_by_species[species] = fauna_by_species.get(species, 0) + 1

            diet_type = entity.diet_type
            if diet_type == DietType.HERBIVORE:
                herbivore_count += 1
            elif diet_type == DietType.CARNIVORE:
                carnivore_count += 1
            elif diet_type == DietType.OMNIVORE:
                omnivore_count += 1

        stats["prey_population"] = herbivore_count
        stats["predator_population"] = carnivore_count + omnivore_count

        # Index de biodiversidad de Shannon-Wiener, no me es muy útil en biomas txikis, ahí es más importante balance
        # predador/presa, pero me parece interesante y además para un futuro, bien.
        total_entities = len(all_entities)
        if total_entities > 0:
            species_counts = list(flora_by_species.values()) + list(fauna_by_species.values())
            proportions = [count / total_entities for count in species_counts]
            shannon_index = -sum(p * np.log(p) for p in proportions if p > 0)
            max_shannon = np.log(len(species_counts)) if len(species_counts) > 0 else 0
            biodiversity_index = shannon_index / max_shannon if max_shannon > 0 else 0
        else:
            biodiversity_index = 0.0

        stats["biodiversity_index"] = biodiversity_index
        stats["unique_species"] = len(flora_by_species) + len(fauna_by_species)

        stress_levels = []

        for entity in all_entities:
            vital_component = entity.get_component(ComponentType.VITAL)
            if vital_component and hasattr(vital_component, 'stress_level'):
                stress_levels.append(vital_component.stress_level)

        if stress_levels:
            stats["avg_stress"] = float(np.mean(stress_levels))
        else:
            stats["avg_stress"] = 0.0

        stats["total_entities"] = stats["num_flora"] + stats["num_fauna"]

        stats["climate_adaptation"] = self.calculate_climate_adaptation(all_entities)
        stats["entity_balance"] = self.calculate_entity_balance(flora, fauna)
        stats["climate_adaptation"] = self.calculate_climate_adaptation(all_entities)

        stats["entity_balance"] = self.calculate_entity_balance(flora, fauna)

        try:
            evolution_trends = self._evolution_registry.analyze_evolution_trends()

            primary_adaptations = evolution_trends.get("primary_adaptations", [])

            for i, adaptation in enumerate(primary_adaptations[:3]):
                trait = adaptation.get('trait', '')
                direction = adaptation.get('direction', '')
                species = adaptation.get('species', '').replace('_', ' ').title()

                trend_text = f"{species}: {trait} {direction}"

                stats[f"evolution_trend_{i + 1}"] = trend_text
                stats[f"evolution_trend_{i + 1}_direction"] = 1 if direction == "increasing" else -1

        except Exception as e:
            self._logger.error(f"Error al analizar tendencias evolutivas: {e}")
            for i in range(3):
                stats[f"evolution_trend_{i + 1}"] = ""
                stats[f"evolution_trend_{i + 1}_direction"] = 0

        return stats

    def _collect_all_attributes(self, all_entities: EntityList) -> Dict[str, List[Any]]:
        attributes: Dict[str, List[Any]] = defaultdict(list)
        self._logger.debug("Collecting attributes")

        for entity in all_entities:
            entity_state: Dict[str, Any] = entity.get_state_fields()
            for attribute, value in entity_state.items():
                attributes[attribute].append(value)

        return attributes

    def _compute_statistics(self, all_attributes: Dict[str, List[Any]]):
        stats: Dict[str, int | float] = {}

        self._logger.debug("Computing state's statistics...")

        if "general" in all_attributes:
            general_data = all_attributes["general"]
            if isinstance(general_data, list) and general_data:
                stress_levels = []
                for item in general_data:
                    if isinstance(item, dict) and "stress_level" in item:
                        stress_levels.append(item["stress_level"])

                if stress_levels:
                    stress_values = np.array(stress_levels, dtype=np.float64)
                    valid_stress = stress_values[~np.isnan(stress_values)]
                    if len(valid_stress) > 0:
                        stats["avg_stress_level"] = float(np.mean(valid_stress))

        if "growth" in all_attributes:
            growth_data = all_attributes["growth"]
            if isinstance(growth_data, list) and growth_data:
                sizes = []
                for item in growth_data:
                    if isinstance(item, dict) and "current_size" in item:
                        sizes.append(item["current_size"])

                if sizes:
                    size_values = np.array(sizes, dtype=np.float64)
                    valid_sizes = size_values[~np.isnan(size_values)]
                    if len(valid_sizes) > 0:
                        stats["avg_size"] = float(np.mean(valid_sizes))

        stats["num_flora"] = len(self._entity_provider.get_flora())
        stats["num_fauna"] = len(self._entity_provider.get_fauna())

        return stats

    def calculate_climate_adaptation(self, entities: EntityList) -> float:
        if not entities:
            return 0.0

        adaptation_scores = []

        climate_averages = self._climate_data_manager.get_current_month_averages()
        current_temperature = climate_averages.get("avg_temperature", 20.0)

        for entity in entities:
            weather_adaptation = entity.get_component(ComponentType.WEATHER_ADAPTATION)
            if not weather_adaptation:
                continue

            optimal_temp = weather_adaptation.optimal_temperature
            temp_diff = abs(current_temperature - optimal_temp)

            if current_temperature < optimal_temp:
                resistance = weather_adaptation.cold_resistance
            else:
                resistance = weather_adaptation.heat_resistance

            # Hice esto al principio, se me había pasado. Repensar mejor.
            max_diff = 30.0
            diff_score = max(0.0, 1.0 - (temp_diff / max_diff))
            resistance_score = resistance

            adaptation_score = max(diff_score, resistance_score)
            adaptation_scores.append(adaptation_score)

        return sum(adaptation_scores) / len(adaptation_scores) if adaptation_scores else 0.0

    def _get_empty_stats(self) -> Dict[str, int | float]:
        return {
            "num_flora": 0,
            "num_fauna": 0,
            "avg_health": 0,
            "avg_age": 0,
            "avg_energy": 0,
            "biodiversity_index": 0,
            "unique_species": 0,
        }

    def calculate_entity_balance(self, flora: EntityList, fauna: EntityList) -> float:
        if not flora and not fauna:
            return 0.0

        flora_by_species = {}
        fauna_by_species = {}

        for entity in flora:
            species = str(entity.get_species())
            flora_by_species[species] = flora_by_species.get(species, 0) + 1

        for entity in fauna:
            species = str(entity.get_species())
            fauna_by_species[species] = fauna_by_species.get(species, 0) + 1

        total_species = len(flora_by_species) + len(fauna_by_species)
        if total_species <= 1:
            return 0.0

        total_entities = len(flora) + len(fauna)
        if total_entities == 0:
            return 0.0

        species_counts = list(flora_by_species.values()) + list(fauna_by_species.values())
        proportions = [count / total_entities for count in species_counts]
        shannon_index = -sum(p * np.log(p) for p in proportions if p > 0)

        max_shannon = np.log(total_species)

        if max_shannon == 0:
            return 0.0

        return shannon_index / max_shannon
