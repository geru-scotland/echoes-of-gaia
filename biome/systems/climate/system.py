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
import random
from logging import Logger
from typing import Dict, Any

from biome.systems.climate.seasons import SeasonSystem
from biome.systems.climate.state import ClimateState
from shared.enums import BiomeType, Season, WeatherEvent
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from utils.loggers import LoggerManager
from utils.normalization.ranges import CLIMATE_RANGES


class ClimateSystem:
    def __init__(self, biome_type: BiomeType, initial_season: Season):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        # TODO: Hacer cambio climático - cada x tiempo, forzar una subida de temperatura general, cada 100 años
        # investigar un poco sobre esto.
        self._biome_type: BiomeType = biome_type

        self._initial_state: Dict[str, Any] = {}
        self._base_environmental_factors: Dict[str, Any] = {}
        self._weather_event_deltas: Dict[WeatherEvent, float] = {}
        self._seasonal_info:  Dict[Season, Dict[str, int | float]] = {}
        self._load_environmental_data()

        self._state: ClimateState = self._initialize_state()
        self._season_system: SeasonSystem = SeasonSystem(initial_season)

        for i in range(360):
            self._season_system.update(handle_new_season=self._handle_new_season)

    def _load_environmental_data(self):
        self._base_environmental_factors = BiomeStore.biomes.get(self._biome_type, {}).get("environmental_factors", {})
        self._weather_event_deltas = BiomeStore.weather_event_deltas
        self._seasonal_info = BiomeStore.biomes.get(self._biome_type, {}).get("seasonal_info", {})

    def _initialize_state(self) -> ClimateState:
        try:
            self._initial_state = {
                "temperature": random.randint(
                    self._base_environmental_factors.get("temperature", {}).get("min", 15),
                    self._base_environmental_factors.get("temperature", {}).get("max", 15)
                ),
                "humidity": random.randint(
                    self._base_environmental_factors.get("humidity", {}).get("min", 15),
                    self._base_environmental_factors.get("humidity", {}).get("max", 15)
                ),
                "precipitation": self._base_environmental_factors.get("precipitation", 15),
                "biomass_density": self._base_environmental_factors.get("biomass_density", 15),
                "fauna_density": self._base_environmental_factors.get("fauna_density", 15),
                "co2_level": self._base_environmental_factors.get("co2_level", 15),
                "atm_pressure": self._base_environmental_factors.get("atm_pressure", 15),
            }
            state: ClimateState = ClimateState(**self._initial_state)
            return state
        except Exception as e:
            self._logger.exception(f"An exception occured when trying to obtain environmental factors: {e}")


    def update(self, weather_event: WeatherEvent = None) -> None:
        # Este método es el que irá con simpy para el Bioma, lo hago público
        # para que el adaptador de Reinforcement pueda invocarlo
        # Aquí, el modelo debe decidir. Una vez entrenado
        # y en inferencia, llamo al agente para que me de la decision.
        # Pero, para entrenamiento al ser cálculos simulados en un entorno
        # aislado, calculo los factores en el adapter.
        self._season_system.update(self._handle_new_season)
        self._handle_weather_event(weather_event)
        self._recalculate_humidity()  # Recalcular humedad después de cambios


    def _handle_new_season(self, season: Season) -> None:
        # por ahora, solo la presión atmosférica cambio
        season_deltas: Dict[str, int|float] = self._seasonal_info.get(season, {}).get("deltas", {})

        if not season_deltas:
            self._logger.error(f"There weren't any seasonal deltas to update the environmental factors.")
            return
        # TODO: Quizá introducir ruido al delta también
        self._state.atm_pressure += season_deltas["atm_pressure"]
        self._logger.debug(f"Atmospheric pressure updated: {self._state.atm_pressure} (delta: {season_deltas['atm_pressure']})")

    def _handle_weather_event(self, weather_event: WeatherEvent):
        # Por ahora solo cambio de temperatura, ya veremos en un futuro.
        # TODO: Incluir ruido, variabilidad aquí
        noise = random.gauss(0, 0.5)
        delta_temp = self._weather_event_deltas[weather_event] + noise

        new_temp = self._state.temperature + delta_temp

        # TODO: tomar de ranges.py
        PHYSICAL_MIN_TEMP = CLIMATE_RANGES["temperature"][0]
        PHYSICAL_MAX_TEMP = CLIMATE_RANGES["temperature"][1]
        self._state.temperature = max(PHYSICAL_MIN_TEMP, min(new_temp, PHYSICAL_MAX_TEMP))

        PHYSICAL_MIN_HUM = CLIMATE_RANGES["humidity"][0]
        PHYSICAL_MAX_HUM = CLIMATE_RANGES["humidity"][1]
        self._state.humidity = max(PHYSICAL_MIN_HUM, min(self._state.humidity, PHYSICAL_MAX_HUM))

    def _recalculate_humidity(self) -> None:
        try:
            base_humidity = self._initial_state.get("humidity", 50)
            current_temp = self._state.temperature
            base_temp = self._initial_state.get("temperature", 20)
            current_pressure = self._state.atm_pressure
            base_pressure = self._initial_state.get("atm_pressure", 1013)

            temp_influence = -0.8  # Hago que la humedad disminuye cuando la temperatura aumenta
            pressure_influence = 0.3  # y humedad aumenta ligeramente con la presión

            temp_effect = temp_influence * (current_temp - base_temp)
            pressure_effect = pressure_influence * ((current_pressure - base_pressure) / 10)

            noise = random.uniform(-3, 3)

            new_humidity = base_humidity + temp_effect + pressure_effect + noise
            # TODO: Coge esto de ranges.py
            new_humidity = max(0, min(100, new_humidity))

            self._state.humidity = new_humidity
            self._logger.debug(
                f"Humidity recalculated: {new_humidity:.2f}% (base: {base_humidity}, temp effect: {temp_effect:.2f}, pressure effect: {pressure_effect:.2f})")

        except Exception as e:
            self._logger.exception(f"Error recalculating humidity: {e}")

    def get_state(self) -> ClimateState:
        return self._state

    def get_current_season(self) -> Season:
        return self._season_system.get_current_season()

    def get_seasonal_deltas(self) -> Dict[str, Any]:
        return self._seasonal_info.get(self.get_current_season(), {}).get("deltas", {})

    def get_seasonal_comfort_range(self) -> Dict[str, Any]:
        return self._seasonal_info.get(self.get_current_season(), {}).get("comfort_range", {})

    @property
    def temperature(self) -> float:
        return self._state.temperature

    @property
    def base_environmental_factors(self) -> Dict[str, int|float]:
        return self._base_environmental_factors

    @property
    def weather_event_deltas(self) -> Dict[WeatherEvent, float]:
        return self._weather_event_deltas

    @property
    def seasonal_info(self) -> Dict[Season, Dict[str, int|float]]:
        return self._seasonal_info

    @property
    def seasonal_detals(self) -> Dict[Season, Dict[str, int|float]]:
        return self._seasonal_info

    @property
    def biome_type(self):
        return self._biome_type