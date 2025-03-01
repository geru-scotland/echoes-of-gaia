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
from typing import List, Dict, Any

from biome.systems.climate.system import ClimateSystem
from research.training.reinforcement.adapter import EnvironmentAdapter
from shared.enums import BiomeType, Season
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ClimateTrainAdapter(EnvironmentAdapter):
    def __init__(self, biome_type: BiomeType):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._climate_system: ClimateSystem = ClimateSystem(biome_type)
        # TODO: Nota, los targets, tienen que conseguirlos el modelo. PERO, dentro de los targets
        # Está también la presión atmosférica. Ésta NO han de conseguirla el modelo, ésta la cambio
        # al entrar en la season - primero de manera abrupta, y quizá en un futuro, de manera gradual
        # Pero la presión atmosférica NO es un target para el modelo realmente.
        self._base_environmental_factors: Dict[str, int | float] = self._climate_system.base_environmental_factors
        self._season_targets: Dict[Season, Dict[str, int|float]] = self._precompute_seasonal_targets()
        self._initial_season: bool = True
        print("TARGETS:")
        print(self.get_season_target(Season.SPRING))
        # Extras, los que quiera simular que no vayan a estar en el system por defecto
        self._fauna: int = 2342432
        self._flora: int = 2342

    def _precompute_seasonal_targets(self) -> Dict[Season, Dict[str, int|float]]:
        try:
            deltas: Dict[Season, Dict[str, int|float]] = self._get_seasonal_deltas()
            targets: Dict[Season, Dict[str, int|float]] = {}

            for season, environmental_factors in deltas.items():
                season_targets: Dict[str, int|float] = {}
                for env_factor, value in environmental_factors.items():
                   season_targets[env_factor] = self._base_environmental_factors[env_factor] + value
                targets[season] = season_targets

            return targets
        except Exception as e:
            self._logger.exception(f"There was an exception when trying to precompute seasonal targets: {e}")

    def _get_seasonal_deltas(self) -> Dict[Season, Dict[str, int|float]]:
        try:
            deltas: Dict[Season, Dict[str, int|float]] = BiomeStore.biomes.get(self._climate_system.biome_type, {}).get("seasonal_deltas", {})
            return deltas
        except Exception as e:
            self._logger.exception(f"An exception occured when trying to obtain seasonal deltas: {e}")

    def get_season_target(self, season: Season):
        # OJO, para la primera season, la inicial, el target seran los base values
        # para el resto si, los detales.

        if self._initial_season:
            self._initial_season = False
            return self._base_environmental_factors

        return self._season_targets.get(season, {})

    def _update_biological_factors(self):
        pass

    def progress_climate(self):
        # Desde el entorno la llamo
        # apply weather effects, update temporal factors (seasons)
        # natural processes (enfriamiento, evaporación, calentamiento global (cambio climático))
        self._update_biological_factors()

    def compute_simulated_humidity(self):
        pass

    def compute_simulated_precipitation(self):
        pass

    def compute_simulated_co2(self):
        pass

    def compute_reward(self):
        pass

    def get_observation(self) -> List[float]:
        pass