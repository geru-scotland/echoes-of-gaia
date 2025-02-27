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

from biome.systems.managers.entity_manager import EntityManager
from shared.strings import Loggers
from shared.types import EntityList
from utils.loggers import LoggerManager


class EntityDataCollector:

    def __init__(self, entity_manager: EntityManager):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.entity_manager: EntityManager = entity_manager

    def collect_data(self) -> Dict[str, int|float]:
        self._logger.debug("EntityDataCollectar has started collecting data...")
        flora, fauna = self.entity_manager.get_entities()

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

    def _compute_statistics(self, all_attributes: Dict[str, Any]):
        stats: Dict[str, int|float] = {}

        self._logger.debug("Computing state's statistics...")
        self._logger.debug(f"All attributes: {all_attributes}")
        for attribute, values in all_attributes.items():
            try:
                numeric_values: np.ndarray = np.array(values, dtype=np.float64)

                # isnan crea nuevo array, con true donde sea nan
                # invierto esos valores bitwise ~
                # indexación booleana en numeric_values[], coge solo los true
                numeric_values = numeric_values[~np.isnan(numeric_values)]

                if len(numeric_values) > 0:
                    stats[f"avg_{attribute}"] = float(np.mean(numeric_values))
            except (TypeError, ValueError):
                # para que no pete si attr no numérico
                continue

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