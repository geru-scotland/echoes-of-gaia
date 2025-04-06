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
from abc import ABC
from typing import Dict, Any

from biome.components.handlers.base import AttributeHandler

from biome.systems.events.event_notifier import EventNotifier
from shared.enums.events import ComponentEvent
from shared.enums.reasons import StressReason


class StressHandler(AttributeHandler, ABC):

    def __init__(self, event_notifier: EventNotifier, lifespan: float = 10.0):
        super().__init__(event_notifier)

        self._stress_level: float = round(0.0000, 4)
        self._max_stress: float = 100.0
        self._lifespan: float = lifespan

    def register_events(self):
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)

    def _handle_stress_update(self, *args, **kwargs):
        new_stress: float = kwargs.get("stress_level", 0.0)
        self._stress_level = new_stress

    def modify_stress(self, delta: float, reason: StressReason):
        self._logger.debug(f"[STRESS DEBUG] - Delta: {delta}")
        self._logger.debug(f"[STRESS DEBUG] - Reason: {reason}")
        self._logger.debug(f"[STRESS DEBUG] - Previous Stress Level: {self._stress_level}")
        self._logger.debug(f"[STRESS DEBUG] - Max Stress Allowed: {self._max_stress}")

        lifespan_factor = 1.0 / math.sqrt(max(1.0, self._lifespan))

        old_stress = self._stress_level
        stress_change = delta * lifespan_factor
        new_stress = max(0.0, min(old_stress + stress_change, self._max_stress))
        rounded_stress = round(new_stress, 4)

        self._logger.debug(f"[STRESS DEBUG] - New Calculated Stress (Before Rounding): {new_stress}")
        self._logger.debug(f"[STRESS DEBUG] - New Stress Level (Rounded): {rounded_stress}")

        if old_stress != rounded_stress:
            self._event_notifier.notify(
                ComponentEvent.STRESS_UPDATED,
                stress_level=rounded_stress,
                stress_delta=rounded_stress - old_stress,
                normalized_stress=rounded_stress / self._max_stress,
                reason=reason
            )

            # if self._stress_level > self._max_stress * 0.9 and not self._is_dormant:
            #     self.request_dormancy(DormancyReason.ENVIRONMENTAL_STRESS, True)
            # elif self._stress_level < self._max_stress * 0.7  and DormancyReason.ENVIRONMENTAL_STRESS in self._dormancy_reasons:
            #     self.request_dormancy(DormancyReason.ENVIRONMENTAL_STRESS, False)


    @property
    def stress_level(self) -> float:
        return self._stress_level

    @property
    def max_stress(self) -> float:
        return self._max_stress

    def get_state(self) -> Dict[str, Any]:
        return {
            "stress_level": self._stress_level,
            "max_stress": self._max_stress
        }