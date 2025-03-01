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
from logging import Logger
from typing import Dict, Any

from biome.systems.climate.seasons import SeasonSystem
from biome.systems.climate.state import ClimateState
from shared.enums import BiomeType, Season
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ClimateSystem:
    def __init__(self, biome_type: BiomeType):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        # TODO: Hacer cambio climático - cada x tiempo, forzar una subida de temperatura general, cada 100 años
        # investigar un poco sobre esto.
        self._biome_type: BiomeType = biome_type
        self._base_environmental_factors: Dict[str, int | float] = {}
        self._season_system: SeasonSystem = SeasonSystem()
        self._state: ClimateState = self._initialize_state()
        self._test_seasons()

    def _test_seasons(self):
        for _ in range(365):
            self._season_system.update()

    def _initialize_state(self) -> ClimateState:
        try:
            self._base_environmental_factors = BiomeStore.biomes.get(self._biome_type, {}).get("environmental_factors", {})
            print(self._base_environmental_factors)
            state: ClimateState = ClimateState(**self._base_environmental_factors)
            return state
        except Exception as e:
            self._logger.exception(f"An exception occured when trying to obtain environmental factors: {e}")

    @property
    def base_environmental_factors(self):
        return self._base_environmental_factors

    @property
    def biome_type(self):
        return self._biome_type