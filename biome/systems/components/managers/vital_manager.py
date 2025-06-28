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

"""
Vital functions management for core entity lifecycle processes.

Controls aging, vitality, and health mechanics for all entities; applies 
non-linear aging patterns and vitality decay based on stress and energy levels.
Handles entity death events and cleanup operations - provides comprehensive
health state tracking throughout entity lifespans.
"""

from logging import Logger

import numpy as np
from simpy import Environment as simpyEnv

from biome.components.physiological.vital import VitalComponent
from biome.systems.components.managers.base import BaseComponentManager
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import StressReason
from shared.enums.strings import Loggers
from shared.enums.thresholds import VitalThresholds
from shared.math.biological import BiologicalGrowthPatterns
from shared.timers import Timers
from utils.loggers import LoggerManager


class VitalComponentManager(BaseComponentManager[VitalComponent]):
    def __init__(self, env: simpyEnv, cleanup_dead_entities: bool = True):
        super().__init__(env)

        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._cleanup_dead_entities: bool = cleanup_dead_entities

        self._age_process = self._env.process(self._update_all_age(Timers.Components.Physiological.AGING))

        self._vitality_stress_process = self._env.process(
            self._update_all_vitality_stress(Timers.Components.Physiological.STRESS_UPDATE))

        self._vitality_process = self._env.process(
            self._update_all_vitality(Timers.Components.Physiological.HEALTH_DECAY))

        self._stress_sync_process = self._env.process(
            self._synchronize_aging_rates(Timers.Components.Physiological.STRESS_UPDATE)
        )

    def _update_all_age(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                ages = np.array([comp.age for comp in active_components])
                aging_rates = np.array([comp.aging_rate for comp in active_components])

                new_ages = ages + timer
                new_biological_ages = new_ages * aging_rates

                for i, component in enumerate(active_components):
                    component.age = new_ages[i]
                    component.biological_age = new_biological_ages[i]
                    component.event_notifier.notify(
                        ComponentEvent.UPDATE_STATE,
                        ComponentType.VITAL,
                        age=component.age,
                        biological_age=component.biological_age
                    )

            yield self._env.timeout(timer)

    def _update_all_vitality_stress(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                vitalities = np.array([comp.vitality for comp in active_components])
                max_vitalities = np.array([comp.max_vitality for comp in active_components])

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
                        component.stress_handler.modify_stress(stress_change, StressReason.CRITICAL_VITALITY)
                        self._logger.debug(
                            f"Vitality is CRITICAL ({vitality_ratios[i]:.2f}). Increasing stress by {stress_change:}."
                        )
                    elif low_mask[i]:
                        stress_change = VitalThresholds.StressChange.LOW
                        component.stress_handler.modify_stress(stress_change, StressReason.LOW_VITALITY)
                        self._logger.debug(
                            f"Vitality is LOW ({vitality_ratios[i]:.2f}). Increasing stress by {stress_change}."
                        )
                    elif excellent_mask[i]:
                        stress_change = VitalThresholds.StressChange.EXCELLENT
                        component.stress_handler.modify_stress(stress_change, StressReason.EXCELLENT_VITALITY)
                        self._logger.debug(
                            f"Vitality is EXCELLENT ({vitality_ratios[i]:.2f}). Reducing stress by {stress_change}."
                        )
                    elif good_mask[i]:
                        stress_change = VitalThresholds.StressChange.GOOD
                        component.stress_handler.modify_stress(stress_change, StressReason.GOOD_VITALITY)
                        self._logger.debug(
                            f"Vitality is GOOD ({vitality_ratios[i]:.2f}). Reducing stress by {stress_change}."
                        )

            yield self._env.timeout(timer)

    def _update_all_vitality(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                biological_ages = np.array([comp.biological_age for comp in active_components])
                lifespans = np.array([comp.lifespan_in_ticks for comp in active_components])
                health_modifiers = np.array([comp.health_modifier for comp in active_components])
                max_vitalities = np.array([comp.max_vitality for comp in active_components])
                current_vitalities = np.array([comp.vitality for comp in active_components])
                stress_levels = np.array([comp.stress_handler.stress_level for comp in active_components])
                max_stress_levels = np.array([comp.stress_handler.max_stress for comp in active_components])
                energy_reserves = np.array([comp.energy_reserves for comp in active_components])
                accumulated_decays = np.array([comp.accumulated_decay for comp in active_components])
                accumulated_stress = np.array([comp.accumulated_stress for comp in active_components])

                completed_lifespan_ratios = np.minimum(1.0, biological_ages / lifespans)
                completed_lifespan_ratios_with_mods = completed_lifespan_ratios * health_modifiers

                max_stress_mask = stress_levels >= max_stress_levels - 0.5
                energy_critical_mask = energy_reserves <= 0.1

                non_linear_aging_progressions = BiologicalGrowthPatterns.gompertz_decay_vectorized(
                    completed_lifespan_ratios_with_mods)

                new_healths = max_vitalities * (1.0 - non_linear_aging_progressions)

                new_healths[energy_critical_mask] -= accumulated_decays[energy_critical_mask]
                new_healths[max_stress_mask] -= accumulated_stress[max_stress_mask]

                LOW_VITALITY_THRESHOLD = 0.10  # 10% de la vitalidad máxima
                LOW_VITALITY_DECAY_RATE = 0.05  # 5% de decay por iteración

                low_vitality_mask = (new_healths / max_vitalities) < LOW_VITALITY_THRESHOLD

                if np.any(low_vitality_mask):
                    # Aplica un decay adicional de 5% a las entidades con baja vitalidad
                    additional_decay = max_vitalities[low_vitality_mask] * LOW_VITALITY_DECAY_RATE
                    new_healths[low_vitality_mask] -= additional_decay

                for i, component in enumerate(active_components):

                    if energy_critical_mask[i]:
                        component.increase_accumulated_decay(1.0)

                    if max_stress_mask[i]:
                        component.increase_accumulated_stress(1.0)

                    if new_healths[i] <= 0.0:
                        component.vitality = 0
                        component.event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            ComponentType.VITAL,
                            vitality=component.vitality
                        )
                        component.event_notifier.notify(ComponentEvent.ENTITY_DEATH, ComponentType.VITAL,
                                                        cleanup_dead_entities=self._cleanup_dead_entities)
                    else:
                        component.vitality = max(0, new_healths[i])
                        component.event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            ComponentType.VITAL,
                            vitality=component.vitality
                        )

                    age_in_years = component.age / float(Timers.Calendar.YEAR)
                    component.vitality_history.append((age_in_years, component.vitality))

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
                comp.stress_handler.stress_level / comp.stress_handler.max_stress
                for comp in active_components
            ])

            low_stress_mask = normalized_stresses <= 0.3
            high_stress_mask = ~low_stress_mask

            new_aging_rates = np.zeros_like(normalized_stresses)

            # sistema que le he pasado a linealg
            # M = np.array([
            #     [1,    0,        0,           0,         0,    0,        0,         0],   # f1(0) = 1
            #     [0,    1,        0,           0,         0,    0,        0,         0],   # f1'(0) = 0
            #     [1,  0.3,   0.3**2,     0.3**3,     0,    0,        0,         0],        # f1(0.3) = 0.85
            #     [0,    1,   2*0.3,   3*0.3**2,     0,    0,        0,         0],         # f1'(0.3) = 0
            #     [0,    0,        0,           0,     1,  0.3,   0.3**2,   0.3**3],        # f2(0.3) = 0.85
            #     [0,    0,        0,           0,     0,    1,   2*0.3,   3*0.3**2],       # f2'(0.3) = 0
            #     [0,    0,        0,           0,     1,    1,        1,         1],       # f2(1) = 1.5
            #     [0,    0,        0,           0,     0,    1,        2,         3]        # f2'(1) = 0
            # ])
            #
            # y = np.array([1, 0, 0.85, 0, 0.85, 0, 1.5, 0])
            
            # GRÁFICA: https://echoes-of-gaia.com/images/figures/hormesis_corregida.png
            if np.any(low_stress_mask):
                low_stress = normalized_stresses[low_stress_mask]
                new_aging_rates[low_stress_mask] = (
                        1.0
                        + 0.0 * low_stress
                        - 5.0 * (low_stress ** 2)
                        + 11.111111111111109 * (low_stress ** 3)
                )

            if np.any(high_stress_mask):
                high_stress = normalized_stresses[high_stress_mask]
                new_aging_rates[high_stress_mask] = (
                        1.3104956268221577
                        - 3.4110787172011685 * high_stress
                        + 7.390670553935865 * (high_stress ** 2)
                        - 3.790087463556854 * (high_stress ** 3)
                )

            for i, component in enumerate(active_components):
                # TODO: No actualizo el state aquí, revisar por qué no lo hice, creo que fué despiste simplemente.
                component.aging_rate = new_aging_rates[i]
                self._logger.debug(
                    f"normalized: {normalized_stresses[i]:.2f} → aging_rate: {component.aging_rate:.2f}"
                )
