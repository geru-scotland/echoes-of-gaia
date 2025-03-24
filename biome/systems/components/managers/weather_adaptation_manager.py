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
from typing import Any, Dict, List

import numpy as np
from simpy import Environment as simpyEnv
from biome.systems.events.event_bus import BiomeEventBus
from shared.enums.enums import WeatherEvent
from shared.enums.events import BiomeEvent, ComponentEvent
from shared.enums.reasons import StressReason
from shared.enums.thresholds import ClimateThresholds


class WeatherAdaptationComponentManager:
    def __init__(self, env: simpyEnv):
        self._env = env
        self._components: Dict[int, Any] = {}
        self._component_ids = set()

        BiomeEventBus.register(BiomeEvent.WEATHER_UPDATE, self._handle_weather_update_for_all)

    def register_component(self, component_id: int, component: Any) -> None:
        self._components[component_id] = component
        self._component_ids.add(component_id)

    def unregister_component(self, component_id: int) -> None:
        if component_id in self._components:
            del self._components[component_id]
            self._component_ids.discard(component_id)

    def _get_active_components(self) -> List[Any]:
        return [comp for comp in self._components.values() if comp._host_alive]

    def _handle_weather_update_for_all(self, *args, **kwargs):
        temperature: float = kwargs.get("temperature", 0.0)
        weather_event: WeatherEvent = kwargs.get("weather_event")

        active_components = self._get_active_components()
        if not active_components:
            return

        optimal_temperatures = np.array([comp._optimal_temperature for comp in active_components])
        deviations = temperature - optimal_temperatures

        cold_mask = deviations < 0
        heat_mask = ~cold_mask

        resistances = np.zeros_like(deviations)
        if np.any(cold_mask):
            resistances[cold_mask] = np.array([comp._cold_resistance for i, comp in enumerate(active_components)
                                               if cold_mask[i]])

        if np.any(heat_mask):
            resistances[heat_mask] = np.array([comp._heat_resistance for i, comp in enumerate(active_components)
                                               if heat_mask[i]])

        complete_resistance_mask = resistances >= 1.0
        partial_resistance_mask = ~complete_resistance_mask

        stress_deltas = np.zeros(len(active_components))

        if np.any(partial_resistance_mask):
            # Hardcodeo, como el max stress delta más abajo, porque todos tienen lo mismo.
            base_sigmas = 15.0
            resistances_partial = resistances[partial_resistance_mask]
            sigmas = base_sigmas / (1.0 - resistances_partial)

            # factor de estrés, uso la fórmula gaussiana
            deviations_partial = deviations[partial_resistance_mask]
            stress_factors = np.exp(-(deviations_partial ** 2) / (2 * sigmas ** 2))

            # Hardcodeo para no recorrer lista de componentes, que tienen todos lo mismo.
            max_stress_deltas = 0.2
            stress_deltas_partial = max_stress_deltas * (1 - stress_factors)

            stress_deltas[partial_resistance_mask] = stress_deltas_partial

        for i, component in enumerate(active_components):
            if stress_deltas[i] > 0.001:
                component._stress_handler.modify_stress(stress_deltas[i], StressReason.TEMPERATURE_EXTREME)
                component._logger.debug(
                    f"Temperature stress delta applied: {stress_deltas[i]:.4f} (Temperature: {temperature}°C)")
            else:
                component._stress_handler.modify_stress(-ClimateThresholds.StressChange.OPTIMAL,
                                                        StressReason.TEMPERATURE_OPTIMAL)
                component._logger.debug(f"Optimal temperature relief applied. (Temperature: {temperature}°C)")

            component._event_notifier.notify(ComponentEvent.WEATHER_UPDATE, temperature=temperature,
                                             weather_event=weather_event)