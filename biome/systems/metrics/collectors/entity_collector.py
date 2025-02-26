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
from typing import Dict, Any

from biome.systems.managers.entity_manager import EntityManager


class EntityDataCollector:

    def __init__(self, entity_manager: EntityManager):
        self.entity_manager: EntityManager = entity_manager

    def collect_data(self) -> Dict[str, Any]:
        flora, fauna = self.entity_manager.get_entities()
        if not flora and not fauna:
            return self._get_empty_stats()

        # stats
        for ent in flora:
            print(ent.get_state_fields())

    def _get_empty_stats(self) -> Dict[str, Any]:
        return {
            "num_flora": 0,
            "num_fauna": 0,
            "avg_health": 0,
            "avg_age": 0,
            "avg_energy": 0,
            "biodiversity_index": 0,
            "unique_species": 0,
        }