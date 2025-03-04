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
import math
from logging import Logger
from typing import List, Dict, Any

import numpy as np

from biome.systems.climate.state import ClimateState
from biome.systems.climate.system import ClimateSystem
from research.training.reinforcement.adapter import EnvironmentAdapter
from shared.enums import BiomeType, Season, WeatherEvent
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ClimateTrainAdapter(EnvironmentAdapter):
    def __init__(self, biome_type: BiomeType):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._biome_type: BiomeType = biome_type
        self._climate_system: ClimateSystem = ClimateSystem(biome_type, initial_season=Season.SPRING)
        self._state: ClimateState = self._climate_system.get_state()
        # TODO: Nota, los targets, tienen que conseguirlos el modelo. PERO, dentro de los targets
        # Está también la presión atmosférica. Ésta NO han de conseguirla el modelo, ésta la cambio
        # al entrar en la season - primero de manera abrupta, y quizá en un futuro, de manera gradual
        # Pero la presión atmosférica NO es un target para el modelo realmente.
        self._base_environmental_factors: Dict[str, int | float] = self._climate_system.base_environmental_factors
        self._season_targets: Dict[Season, Dict[str, int|float]] = self._precompute_seasonal_targets()
        self._initial_season: bool = True
        # Extras, los que quiera simular que no vayan a estar en el system por defecto
        self._fauna: int = 2342432
        self._flora: int = 2342

    def _precompute_seasonal_targets(self) -> Dict[Season, Dict[str, int|float]]:
        try:
            deltas: Dict[Season, Dict[str, int|float]] = self._climate_system.seasonal_deltas
            targets: Dict[Season, Dict[str, int|float]] = {}

            for season, environmental_factors in deltas.items():
                season_targets: Dict[str, int|float] = {}
                for env_factor, value in environmental_factors.items():
                   season_targets[env_factor] = self._base_environmental_factors[env_factor] + value
                targets[season] = season_targets

            return targets
        except Exception as e:
            self._logger.exception(f"There was an exception when trying to precompute seasonal targets: {e}")



    def get_season_target(self, season: Season):
        # OJO, para la primera season, la inicial, el target seran los base values
        # para el resto si, los detales.

        if self._initial_season:
            self._initial_season = False
            return self._base_environmental_factors

        return self._season_targets.get(season, {})

    def _update_biological_factors(self):
        pass

    def progress_climate(self, weather_event_idx: int):
        # Desde el entorno la llamo
        # apply weather effects, update temporal factors (seasons)
        # natural processes (enfriamiento, evaporación, calentamiento global (cambio climático))

        # Con el weather event, yo sé qué cómo modificar la temperatura (prepara un dict)
        weather_event: WeatherEvent = WeatherEvent(list(WeatherEvent)[weather_event_idx])
        self._update_biological_factors()
        self._climate_system.update(weather_event)


    def compute_simulated_humidity(self):
        pass

    def compute_simulated_precipitation(self):
        pass

    def compute_simulated_co2(self):
        pass

    def compute_reward(self) -> float:
        def seasonal_climate_fidelity() -> float:
            """
            Utilizo decaimiento exponencial negativo. Ahora tomo temperatura
            solo como target, más tarde ya tomaré humidy y precipitation
            """
            targets: Dict[str, int|float] = self._season_targets[self._climate_system.get_current_season()]
            temperature_diff: float = abs(self._state.temperature - targets["temperature"])
            scale_factor: float = 0.1
            # Función exponencial negativa: me da 1.0 cuando la diferencia es 0,
            # y tiende a 0 conforme aumenta la diferencia
            proximity__score: float = math.exp(-scale_factor * temperature_diff)
            base_reward: float = 10.0
            return base_reward * proximity__score

        def extreme_condition_penalty() -> float:
            too_hot: float = 46.0
            too_cold: float = -27.0

            if self._state.temperature > too_hot:
                return -5.0 * (self._state.temperature - too_hot) / 10.0
            elif self._state.temperature < too_cold:
                return -5.0 * (too_cold - self._state.temperature) / 10.0
            return 0.0

        def climate_appropiateness() -> float:
            pass

        def smooth_transition() -> float:
            pass

        return seasonal_climate_fidelity() + extreme_condition_penalty()

    def get_observation(self) -> Dict[str, Any]:
        biome_idx: int = list(BiomeType).index(self._biome_type)
        season_idx: int = list(Season).index(self._climate_system.get_current_season())
        normalized_temp: float = self.normalize_temperature(self._state.temperature)
        normalized_pressure: float = self.normalize_pressure(self._state.atm_pressure)
        return {
            "temperature": np.array([normalized_temp], dtype=np.float32),
            "atm_pressure": np.array([normalized_pressure], dtype=np.float32),
            "biome_type": biome_idx,
            "season": season_idx
        }

    def normalize_temperature(self, temp: float) -> float:
        """ TODO: Pasar los valores extremos a config, sin falta"""
        return (temp - (-30)) / (50 - (-30))

    def normalize_pressure(self, pressure: float) -> float:
        return (pressure - 300) / (1200 - 300)