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
import copy
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
from utils.normalization.normalizer import climate_normalizer


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
        self._initial_season: bool = True
        self._previous_temp = self._state.temperature
        self._previous_humidity = self._state.humidity
        # Extras, los que quiera simular que no vayan a estar en el system por defecto
        self._fauna: int = 2342432
        self._flora: int = 2342

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

    def compute_reward(self, action) -> float:
        comfort_range = self._climate_system.get_seasonal_comfort_range()
        current_temp = self._state.temperature
        current_humidity = self._state.humidity

        # Nueva sintaxis que no conocia, desempaquetar en ordel de campos
        # del json con values
        temp_min, temp_max = comfort_range["temperature"].values()
        humidity_min, humidity_max = comfort_range["humidity"].values()

        temp_in_range = temp_min <= current_temp <= temp_max
        humidity_in_range = humidity_min <= current_humidity <= humidity_max
        self._logger.info(
            f"Current Temperature: {current_temp}, Min: {temp_min}, Max: {temp_max}, In Range: {temp_in_range}")
        self._logger.info(
            f"Current Humidity: {current_humidity}, Min: {humidity_min}, Max: {humidity_max}, In Range: {humidity_in_range}")
        self._logger.info(f"ACTION WAS: {action}")

        base_reward = 0.0
        if temp_in_range:
            base_reward += 8.0
        if humidity_in_range:
            base_reward += 4.0

        if not temp_in_range:
            distance = min(abs(current_temp - temp_min), abs(current_temp - temp_max))
            base_reward -= distance * 0.25

        if not humidity_in_range:
            distance = min(abs(current_humidity - humidity_min), abs(current_humidity - humidity_max))
            base_reward -= distance * 0.2

        extreme_penalty = 0.0
        if current_temp > 45 or current_temp < -25:
            extreme_penalty -= 5.0

        if current_humidity > 95 or current_humidity < 5:
            extreme_penalty -= 5.0

        total_reward = base_reward + extreme_penalty

        self._logger.info(
            f"Rewards - Base: {base_reward:.2f}, "
            f"Extreme penalty: {extreme_penalty:.2f}, "
            f"Total: {total_reward:.2f}"
        )

        return total_reward

    def get_observation(self) -> Dict[str, Any]:
        biome_idx: int = list(BiomeType).index(self._biome_type)
        season_idx: int = list(Season).index(self._climate_system.get_current_season())
        normalized_temp: float = climate_normalizer.normalize("temperature", self._state.temperature)
        normalized_humidity: float = climate_normalizer.normalize("humidity", self._state.humidity)
        normalized_pressure: float = climate_normalizer.normalize("atm_pressure", self._state.atm_pressure)
        return {
            "temperature": np.array([normalized_temp], dtype=np.float32),
            "humidity": np.array([normalized_humidity], dtype=np.float32),
            "atm_pressure": np.array([normalized_pressure], dtype=np.float32),
            "biome_type": biome_idx,
            "season": season_idx
        }