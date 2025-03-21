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

from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.strings import Loggers
from shared.types import EntityList
from utils.loggers import LoggerManager


class EntityDataCollector:

    def __init__(self, entity_provider: EntityProvider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._entity_provider: EntityProvider = entity_provider

    def collect_data(self) -> Dict[str, int|float]:
        self._logger.debug("EntityDataCollectar has started collecting data...")
        flora, fauna = self._entity_provider.get_entities(only_alive=True)

        if not flora and not fauna:
            return self._get_empty_stats()

        all_entities: EntityList = flora + fauna
        all_attributes: Dict[str, List[Any]] = self._collect_all_attributes(all_entities)
        stats: Dict[str, int|float] = self._compute_statistics(all_attributes)
        stats["num_flora"] = len(flora)
        stats["num_fauna"] = len(fauna)
        self._logger.debug(f"Final stats: {stats}")

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


    def _get_empty_stats(self) -> Dict[str, int|float]:
        return {
            "num_flora": 0,
            "num_fauna": 0,
            "avg_health": 0,
            "avg_age": 0,
            "avg_energy": 0,
            "biodiversity_index": 0,
            "unique_species": 0,
        }