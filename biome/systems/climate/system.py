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

from biome.services.climate_service import ClimateService
from biome.systems.climate.seasons import SeasonSystem
from biome.systems.climate.state import ClimateState
from shared.enums.enums import BiomeType, Season, WeatherEvent
from shared.stores.biome_store import BiomeStore
from shared.enums.strings import Loggers
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
        self._weather_event_effects: Dict[WeatherEvent, Dict[Any]] = {}
        self._seasonal_info:  Dict[Season, Dict[str, int | float]] = {}
        self._load_environmental_data()

        self._state: ClimateState = self._initialize_state()
        ClimateService.init_service(self._state)

        self._season_system: SeasonSystem = SeasonSystem(initial_season)

        for i in range(360):
            self._season_system.update(handle_new_season=self._handle_new_season)

    def _load_environmental_data(self):
        self._base_environmental_factors = BiomeStore.biomes.get(self._biome_type, {}).get("environmental_factors", {})
        self._weather_event_effects = BiomeStore.weather_event_effects
        self._seasonal_info = BiomeStore.biomes.get(self._biome_type, {}).get("seasonal_info", {})

    def _initialize_state(self) -> ClimateState:
        try:
            self._initial_state = {
                "temperature": random.randint(
                    self._base_environmental_factors.get("temperature", {}).get("min", 15),
                    self._base_environmental_factors.get("temperature", {}).get("max", 15)
                ),
                "humidity": random.randint(
                    self._base_environmental_factors.get("humidity", {}).get("min", 20),
                    self._base_environmental_factors.get("humidity", {}).get("max", 20)
                ),
                "precipitation": random.randint(
                    self._base_environmental_factors.get("precipitation", {}).get("min", 0),
                    self._base_environmental_factors.get("precipitation", {}).get("max", 0)
                ),
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
        self._logger.debug(f"Handling weather event: {weather_event}")
        self._logger.debug(f"State previously: {self._state}")

        min_effect_temp, max_effect_temp = self._weather_event_effects[weather_event]["temperature"].values()
        min_effect_hum, max_effect_hum = self._weather_event_effects[weather_event]["humidity"].values()
        min_effect_prec, max_effect_prec = self._weather_event_effects[weather_event]["precipitation"].values()

        mod_temperature: float = round(random.uniform(min_effect_temp, max_effect_temp), 1)
        mod_precipitation: float = round(random.uniform(min_effect_prec, max_effect_prec), 1)
        mod_humidity: float = round(random.uniform(min_effect_hum, max_effect_hum), 1)

        self._logger.debug(f"Generated values - Temperature: {mod_temperature}")
        self._logger.debug(f"Generated values - Precipitation: {mod_precipitation}")
        self._logger.debug(f"Generated values - Humidity: {mod_humidity}")

        # TODO: tomar de ranges.py
        PHYSICAL_MIN_TEMP = CLIMATE_RANGES["temperature"][0]
        PHYSICAL_MAX_TEMP = CLIMATE_RANGES["temperature"][1]
        self._logger.debug(
            f"Temperature - PHYSICAL_MIN: {PHYSICAL_MIN_TEMP}, PHYSICAL_MAX: {PHYSICAL_MAX_TEMP}, "
            f"MOD: {mod_temperature}, CURRENT_STATE: {self._state.temperature}"
        )
        self._state.temperature = round(max(PHYSICAL_MIN_TEMP,
                                      min(self._state.temperature + mod_temperature, PHYSICAL_MAX_TEMP)), 1)

        PHYSICAL_MIN_HUM = CLIMATE_RANGES["humidity"][0]
        PHYSICAL_MAX_HUM = CLIMATE_RANGES["humidity"][1]
        self._logger.debug(
            f"Humidity - PHYSICAL_MIN: {PHYSICAL_MIN_HUM}, PHYSICAL_MAX: {PHYSICAL_MAX_HUM}, "
            f"MOD: {mod_humidity}, CURRENT_STATE: {self._state.humidity}"
        )
        self._state.humidity = round(max(PHYSICAL_MIN_HUM, min(self._state.humidity + mod_humidity, PHYSICAL_MAX_HUM)), 1)

        PHYSICAL_MIN_PREC = CLIMATE_RANGES["precipitation"][0]
        PHYSICAL_MAX_PREC = CLIMATE_RANGES["precipitation"][1]
        self._logger.debug(
            f"Precipitation - PHYSICAL_MIN: {PHYSICAL_MIN_PREC}, PHYSICAL_MAX: {PHYSICAL_MAX_PREC}, "
            f"MOD: {mod_precipitation}, CURRENT_STATE: {self._state.precipitation}"
        )
        self._state.precipitation = round(max(PHYSICAL_MIN_PREC,
                                        min(self._state.precipitation + mod_precipitation, PHYSICAL_MAX_PREC)), 1)

        # TODO: Establecer umbrales para temperatura, hum y prec, y
        # triggear aquí para los que se hayan subscrito
        self._logger.debug(f"State AFTER: {self._state}")

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
    def weather_event_effects(self) -> Dict[WeatherEvent, dict]:
        return self._weather_event_effects

    @property
    def seasonal_info(self) -> Dict[Season, Dict[str, int|float]]:
        return self._seasonal_info

    @property
    def seasonal_detals(self) -> Dict[Season, Dict[str, int|float]]:
        return self._seasonal_info

    @property
    def biome_type(self):
        return self._biome_type