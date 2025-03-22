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
from typing import Dict, Any

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType, WeatherEvent
from shared.enums.events import ComponentEvent
from shared.enums.reasons import StressReason
from shared.enums.thresholds import ClimateThresholds
from shared.timers import Timers


class WeatherAdaptationComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 cold_resistance: float = 1.0,
                 heat_resistance: float = 1.0,
                 optimal_temperature: float = 22.0,
                 base_sigma: float = 15.0,
                 max_stress_delta: float = 0.2):

        self._stress_handler: StressHandler = StressHandler(event_notifier)
        super().__init__(env, ComponentType.WEATHER_ADAPTATION, event_notifier, lifespan)

        self._cold_resistance: float = cold_resistance
        self._heat_resistance: float = heat_resistance

        self._optimal_temperature: float = optimal_temperature

        self._base_sigma: float = base_sigma
        self._logger.warning(f"BASE SIGMA {self._base_sigma}")
        self._max_stress_delta = max_stress_delta

        self._temperature_exposure_history: list = []

        self._logger.warning(f"TemperatureAdaptationComponent initialized: Cold resistance={self._cold_resistance}, "
                           f"Heat resistance={self._heat_resistance}"
                           f"Optimal temperature={self._optimal_temperature}")

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._event_notifier.register(ComponentEvent.WEATHER_UPDATE, self._handle_weather_update)

    def _handle_weather_update(self, *args, **kwargs):
        temperature: float = kwargs.get("temperature", 0.0)
        weather_event: WeatherEvent = kwargs.get("weather_event")

        stress_delta = self._calculate_stress_delta(temperature)

        if stress_delta > 0.001:
            self._stress_handler.modify_stress(stress_delta, StressReason.TEMPERATURE_EXTREME)
            self._logger.debug(f"Temperature stress delta applied: {stress_delta:.4f} (Temperature: {temperature}°C)")
        else:
            self._stress_handler.modify_stress(-ClimateThresholds.StressChange.OPTIMAL, StressReason.TEMPERATURE_OPTIMAL)
            self._logger.debug(f"Optimal temperature relief applied. (Temperature: {temperature}°C)")

    def _calculate_stress_delta(self, temperature: float) -> float:
        deviation = temperature - self._optimal_temperature

        resistance = self._cold_resistance if deviation < 0 else self._heat_resistance

        if resistance >= 1.0:
            return 0.0

        sigma = self._base_sigma / (1.0 - resistance)

        stress_factor = math.exp(-(deviation ** 2) / (2 * sigma ** 2))
        stress_delta = self._max_stress_delta * (1 - stress_factor)

        return stress_delta

    def get_state(self) -> Dict[str, Any]:
        return {
            "cold_resistance": self._cold_resistance,
            "heat_resistance": self._heat_resistance,
            "optimal_temperature": self._optimal_temperature,
        }

    @property
    def cold_resistance(self) -> float:
        return self._cold_resistance

    @property
    def heat_resistance(self) -> float:
        return self._heat_resistance

    @property
    def optimal_temperature(self) -> float:
        return self._optimal_temperature

