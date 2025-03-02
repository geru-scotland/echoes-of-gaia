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
from shared.enums import BiomeType, Season, WeatherEvent
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ClimateSystem:
    def __init__(self, biome_type: BiomeType, initial_season: Season):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        # TODO: Hacer cambio climático - cada x tiempo, forzar una subida de temperatura general, cada 100 años
        # investigar un poco sobre esto.
        self._biome_type: BiomeType = biome_type

        self._base_environmental_factors: Dict[str, int | float] = {}
        self._weather_event_deltas: Dict[WeatherEvent, float] = {}
        self._seasonal_deltas:  Dict[Season, Dict[str, int | float]] = {}
        self._load_environmental_data()

        self._state: ClimateState = self._initialize_state()
        self._season_system: SeasonSystem = SeasonSystem(initial_season)

        for i in range(360):
            self._season_system.update(handle_new_season=self._handle_new_season)

    def _load_environmental_data(self):
        self._base_environmental_factors = BiomeStore.biomes.get(self._biome_type, {}).get("environmental_factors", {})
        self._weather_event_deltas = BiomeStore.biomes.get("weather_event_deltas", {})
        self._seasonal_deltas = BiomeStore.biomes.get(self._biome_type, {}).get("seasonal_deltas", {})

    def _initialize_state(self) -> ClimateState:
        try:
            state: ClimateState = ClimateState(**self._base_environmental_factors)
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
        season_deltas: Dict[str, int|float] = self._seasonal_deltas.get(season, {})

        if not season_deltas:
            self._logger.error(f"There weren't any seasonal deltas to update the environmental factors.")
            return

        self._state.atm_pressure += season_deltas["atm_pressure"]
        self._logger.debug(f"Atmospheric pressure updated: {self._state.atm_pressure} (delta: {season_deltas['atm_pressure']})")

    def _handle_weather_event(self, weather_event: WeatherEvent):
        # Por ahora solo cambio de temperatura, ya veremos en un futuro.
        if weather_event:
            self._state.temperature += self._weather_event_deltas[weather_event]

    def get_state(self) -> ClimateState:
        return self._state

    def get_current_season(self)-> Season:
        return self._season_system.get_current_season()

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
    def seasonal_deltas(self) -> Dict[Season, Dict[str, int|float]]:
        return self._seasonal_deltas

    @property
    def biome_type(self):
        return self._biome_type