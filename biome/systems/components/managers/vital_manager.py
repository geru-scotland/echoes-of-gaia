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
import numpy as np
from typing import Dict, List, Set, Optional, Any
from simpy import Environment as simpyEnv

from shared.enums.events import ComponentEvent
from shared.enums.enums import ComponentType
from shared.enums.thresholds import VitalThresholds
from shared.enums.reasons import StressReason
from shared.timers import Timers
from shared.math.biological import BiologicalGrowthPatterns


class VitalComponentManager:
    def __init__(self, env: simpyEnv):
        self._env = env
        self._components: Dict[int, Any] = {}
        self._component_ids = set()

        self._age_process = self._env.process(self._update_all_age(Timers.Compoments.Physiological.AGING))
        self._vitality_stress_process = self._env.process(
            self._update_all_vitality_stress(Timers.Compoments.Physiological.STRESS_UPDATE))

        self._vitality_process = self._env.process(
            self._update_all_vitality(Timers.Compoments.Physiological.HEALTH_DECAY))

        self._stress_sync_process = self._env.process(
            self._synchronize_aging_rates(Timers.Compoments.Physiological.STRESS_UPDATE)
        )

    def register_component(self, component_id: int, component: Any) -> None:
        self._components[component_id] = component
        self._component_ids.add(component_id)

    def unregister_component(self, component_id: int) -> None:
        if component_id in self._components:
            del self._components[component_id]
            self._component_ids.discard(component_id)

    def _get_active_components(self) -> List[Any]:
        return [comp for comp in self._components.values() if comp._host_alive]

    def _get_non_dormant_components(self) -> List[Any]:
        return [comp for comp in self._components.values()
                if comp._host_alive and not comp._is_dormant]

    def _update_all_age(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                ages = np.array([comp._age for comp in active_components])
                aging_rates = np.array([comp._aging_rate for comp in active_components])

                new_ages = ages + timer
                new_biological_ages = new_ages * aging_rates

                for i, component in enumerate(active_components):
                    component._age = new_ages[i]
                    component._biological_age = new_biological_ages[i]
                    component._event_notifier.notify(
                        ComponentEvent.UPDATE_STATE,
                        ComponentType.VITAL,
                        age=component._age,
                        biological_age=component._biological_age
                    )

            yield self._env.timeout(timer)

    def _update_all_vitality_stress(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                vitalities = np.array([comp._vitality for comp in active_components])
                max_vitalities = np.array([comp._max_vitality for comp in active_components])

                vitality_ratios = vitalities / max_vitalities

                critical_mask = vitality_ratios < VitalThresholds.Health.CRITICAL
                low_mask = (vitality_ratios >= VitalThresholds.Health.CRITICAL) & (
                            vitality_ratios < VitalThresholds.Health.LOW)
                excellent_mask = vitality_ratios > VitalThresholds.Health.EXCELLENT
                good_mask = (vitality_ratios <= VitalThresholds.Health.EXCELLENT) & (
                            vitality_ratios > VitalThresholds.Health.GOOD)

                for i, component in enumerate(active_components):
                    if critical_mask[i]:
                        stress_change = VitalThresholds.StressChange.CRITICAL
                        component._stress_handler.modify_stress(stress_change, StressReason.CRITICAL_VITALITY)
                        component._logger.debug(
                            f"Vitality is CRITICAL ({vitality_ratios[i]:.2f}). Increasing stress by {stress_change:}."
                        )
                    elif low_mask[i]:
                        stress_change = VitalThresholds.StressChange.LOW
                        component._stress_handler.modify_stress(stress_change, StressReason.LOW_VITALITY)
                        component._logger.debug(
                            f"Vitality is LOW ({vitality_ratios[i]:.2f}). Increasing stress by {stress_change}."
                        )
                    elif excellent_mask[i]:
                        stress_change = VitalThresholds.StressChange.EXCELLENT
                        component._stress_handler.modify_stress(stress_change, StressReason.EXCELLENT_VITALITY)
                        component._logger.debug(
                            f"Vitality is EXCELLENT ({vitality_ratios[i]:.2f}). Reducing stress by {stress_change}."
                        )
                    elif good_mask[i]:
                        stress_change = VitalThresholds.StressChange.GOOD
                        component._stress_handler.modify_stress(stress_change, StressReason.GOOD_VITALITY)
                        component._logger.debug(
                            f"Vitality is GOOD ({vitality_ratios[i]:.2f}). Reducing stress by {stress_change}."
                        )

            yield self._env.timeout(timer)

    def _update_all_vitality(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_non_dormant_components()

            if active_components:
                biological_ages = np.array([comp._biological_age for comp in active_components])
                lifespans = np.array([comp._lifespan_in_ticks for comp in active_components])
                health_modifiers = np.array([comp._health_modifier for comp in active_components])
                max_vitalities = np.array([comp._max_vitality for comp in active_components])
                stress_levels = np.array([comp._stress_handler.stress_level for comp in active_components])
                max_stress_levels = np.array([comp._stress_handler.max_stress for comp in active_components])

                completed_lifespan_ratios = np.minimum(1.0, biological_ages / lifespans)
                completed_lifespan_ratios_with_mods = completed_lifespan_ratios * health_modifiers

                max_stress_mask = stress_levels >= max_stress_levels

                non_linear_aging_progressions = np.zeros_like(completed_lifespan_ratios_with_mods)
                non_linear_aging_progressions = BiologicalGrowthPatterns.gompertz_decay_vectorized(completed_lifespan_ratios_with_mods)

                new_healths = max_vitalities * (1.0 - non_linear_aging_progressions)
                new_healths[max_stress_mask] -= max_vitalities[max_stress_mask] * 0.05

                for i, component in enumerate(active_components):
                    component._logger.debug(
                        f"[Vitality Update | DEBUG:Tick={component._env.now}] "
                        f"Stress level={component._stress_handler.stress_level:4f}]"
                        f"Age: {component._age} - Biological Age: {component._biological_age}, Lifespan: {component._lifespan_in_ticks}, "
                        f"Aging rate: {component._aging_rate} "
                        f"Completed Ratio: {completed_lifespan_ratios[i]:.2f}, Aging Progression: {non_linear_aging_progressions[i]:.2f}, "
                        f"Health Modifier: {component._health_modifier} "
                        f"New Health: {new_healths[i]:.2f}, Vitality: {component._vitality}"
                    )

                    if new_healths[i] <= 0.0 :
                        component._vitality = 0
                        component._event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            ComponentType.VITAL,
                            vitality=component._vitality
                        )
                        component._event_notifier.notify(ComponentEvent.ENTITY_DEATH, ComponentType.VITAL)
                    else:
                        component._vitality = max(0, new_healths[i])
                        component._event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            ComponentType.VITAL,
                            vitality=component._vitality
                        )

                    age_in_years = component._age / float(Timers.Calendar.YEAR)
                    component._vitality_history.append((age_in_years, component._vitality))


            yield self._env.timeout(timer)

    def _synchronize_aging_rates(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            self.handle_all_stress_updates()
            yield self._env.timeout(timer)

    def handle_all_stress_updates(self):
        active_components = self._get_active_components()

        if active_components:
            normalized_stresses = np.array([
                comp._stress_handler.stress_level / comp._stress_handler.max_stress
                for comp in active_components
            ])

            low_stress_mask = normalized_stresses <= 0.2
            high_stress_mask = ~low_stress_mask

            new_aging_rates = np.zeros_like(normalized_stresses)

            if np.any(low_stress_mask):
                low_stress = normalized_stresses[low_stress_mask]
                new_aging_rates[low_stress_mask] = (
                        1 - 11.25 * (low_stress ** 2) + 37.5 * (low_stress ** 3)
                )

            if np.any(high_stress_mask):
                high_stress = normalized_stresses[high_stress_mask]
                new_aging_rates[high_stress_mask] = (
                        0.9921875 - 1.5234375 * high_stress +
                        4.5703125 * (high_stress ** 2) -
                        2.5390625 * (high_stress ** 3)
                )

            for i, component in enumerate(active_components):
                component._aging_rate = new_aging_rates[i]
                component._logger.debug(
                    f"normalized: {normalized_stresses[i]:.2f} → aging_rate: {component._aging_rate:.2f}"
                )