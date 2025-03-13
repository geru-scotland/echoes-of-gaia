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
from typing import Optional, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from simpy import Environment as simpyEnv
from biome.components.base.component import FloraComponent
from shared.math.biological import BiologicalGrowthPatterns
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.enums.thresholds import VitalThresholds
from shared.timers import Timers


class VitalComponent(FloraComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float = 5.0,
                 vitality: float = 100.0, max_vitality: float = 100.0, age: float = 0.0, aging_rate: float = 1.0,
                 dormancy_threshold: float = 25.0):

        super().__init__( env, ComponentType.VITAL, event_notifier)
        self._lifespan_in_days: int = int((lifespan * float(Timers.Calendar.YEAR)) / float(Timers.Calendar.DAY))
        self._vitality: float = round(vitality, 2)
        self._max_vitality: float = round(max_vitality, 2)
        self._age: float = age
        self._aging_rate: float = aging_rate
        self._biological_age: float = self._age * aging_rate
        self._dormancy_threshold: float = round(dormancy_threshold, 2)

        self._health_modifier: float = 1.0
        self._vitality_history: List[Tuple[float, float]] = []  # (age, vitality)
        self._logger.debug(f"Vital component initialized: Health={self._vitality}/{self._max_vitality}, "
                           f"Age={self._age}")

        self._env.process(self._update_age(Timers.Compoments.Physiological.AGING))
        self._env.process(self._update_vitality_stress(Timers.Compoments.Physiological.STRESS_UPDATE))
        self._env.process(self._update_vitality(Timers.Compoments.Physiological.HEALTH_DECAY))

    def _register_events(self):
        super()._register_events()

    def _update_age(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            self._age =  timer / Timers.Calendar.DAY
            self._biological_age = self._age * self._aging_rate
            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, VitalComponent, age=self._age,
                                        biological_age=self._biological_age)

            yield self._env.timeout(timer)

    def _update_vitality_stress(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            vitality_ratio = self._vitality / self._max_vitality

            if vitality_ratio < VitalThresholds.Health.CRITICAL:
                stress_change = VitalThresholds.StressChange.CRITICAL / 10.0
                self.modify_stress(stress_change, StressReason.CRITICAL_VITALITY)
                self._logger.debug(
                    f"Vitality is CRITICAL ({vitality_ratio:.2f}). Increasing stress by {stress_change:.4f}.")

            elif vitality_ratio < VitalThresholds.Health.LOW:
                stress_change = VitalThresholds.StressChange.LOW / 10.0
                self.modify_stress(stress_change, StressReason.LOW_VITALITY)
                self._logger.debug(
                    f"Vitality is LOW ({vitality_ratio:.2f}). Increasing stress by {stress_change:.4f}.")

            elif vitality_ratio > VitalThresholds.Health.EXCELLENT:
                stress_change = VitalThresholds.StressChange.EXCELLENT / 10.0
                self.modify_stress(stress_change, StressReason.EXCELLENT_VITALITY)
                self._logger.debug(
                    f"Vitality is EXCELLENT ({vitality_ratio:.2f}). Reducing stress by {stress_change:.4f}.")

            elif vitality_ratio > VitalThresholds.Health.GOOD:
                stress_change = VitalThresholds.StressChange.GOOD / 10.0
                self.modify_stress(stress_change, StressReason.GOOD_VITALITY)
                self._logger.debug(
                    f"Vitality is GOOD ({vitality_ratio:.2f}). Reducing stress by {stress_change:.4f}.")

            yield self._env.timeout(timer)

    def _update_vitality(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:

            if not self._is_dormant:
                completed_lifespan_ratio: float = min(1.0, self._biological_age / self._lifespan_in_days)
                completed_lifespan_ratio_with_mods: float = completed_lifespan_ratio * self._health_modifier
                non_linear_aging_progression = BiologicalGrowthPatterns.gompertz_decay(completed_lifespan_ratio_with_mods)

                new_health = self._max_vitality * (1.0 - non_linear_aging_progression)
                self._logger.debug(
                    f"[Vitality Update | DEBUG:Tick={self._env.now}] "
                    f"Stress level={self._stress_level:4f}]"
                    f"Age: {self._age} - Biological Age: {self._biological_age}, Lifespan: {self._lifespan_in_days}, "
                    f"Aging rate: {self._aging_rate} "
                    f"Completed Ratio: {completed_lifespan_ratio:.2f}, Aging Progression: {non_linear_aging_progression:.2f}, "
                    f"Health Modifier: {self._health_modifier} "
                    f"New Health: {new_health:.2f}, Vitality: {self._vitality}"
                )

                self._vitality = max(0, new_health)
                self._event_notifier.notify(ComponentEvent.UPDATE_STATE, VitalComponent, health=self._vitality)
                age_in_years = self._age / 365.0
                self._vitality_history.append((age_in_years, self._vitality))

                # TODO: Gestionar en entity, pensar bien la logíca del bloque general de updates
                # si entra en dormancy.
                if self._vitality <= self._dormancy_threshold and not self._is_dormant:
                    self.request_dormancy(DormancyReason.LOW_VITALITY, True)
                elif self._vitality > self._dormancy_threshold and self._is_dormant:
                    self.request_dormancy(DormancyReason.LOW_VITALITY, False)

                # TODO: Mejorar umbral, con Gompertz me hace falta
                if self._vitality <= 0.0018:
                    # TODO: Death. Gestionar en entity
                    self._event_notifier.notify(ComponentEvent.ENTITY_DEATH, VitalComponent)
                    break

            yield self._env.timeout(timer)

    def _handle_stress_update(self, *args, **kwargs):
        super()._handle_stress_update(*args, **kwargs)

        normalized_stress: float = kwargs.get("normalized_stress", 0.0)

        if normalized_stress <= 0.3:
            self._aging_rate = (200 / 9) * normalized_stress ** 3 - 10 * normalized_stress ** 2 + 1
        else:
            self._aging_rate = -22.02 * normalized_stress ** 3 + 36.45 * normalized_stress ** 2 - 15.93 * normalized_stress + 2.79

        self._logger.debug(
            f"normalized: {normalized_stress:.2f} → aging_rate: {self._aging_rate:.2f}")

    def apply_damage(self, amount: float) -> None:
        # Hervíboros comiendo, tiemps extremos...etc
        self._vitality -= amount * self._biological_age
        self._vitality = max(0.0, self._vitality)

        self._event_notifier.notify(ComponentEvent.UPDATE_STATE, VitalComponent, health=self._vitality)

        if self._vitality <= 0:
            self._event_notifier.notify(ComponentEvent.ENTITY_DEATH, VitalComponent)

    def apply_healing(self, amount: float) -> None:
        # clima favorable, que aplique heals
        self._vitality += amount * (1 - self._biological_age)
        self._vitality = min(self._vitality, self._max_vitality)

        self._event_notifier.notify(ComponentEvent.UPDATE_STATE, VitalComponent, health=self._vitality)

    def set_health_modifier(self, modifier: float) -> None:
        self._health_modifier = max(0.0, modifier)

    def plot_vitality_curve(self) -> None:
        if not self._vitality_history:
            self._logger.warning("No hay datos de vitalidad")
            return

        ages, vitality_values = zip(*self._vitality_history)

        plt.figure(figsize=(10, 6))

        plt.plot(ages, vitality_values, 'o-', color='blue', markersize=4, label='Vitalidad real')

        plt.axhline(y=self._dormancy_threshold, color='orange', linestyle='--',
                    label=f'Umbral de dormancia ({self._dormancy_threshold})')

        if len(ages) > 5:
            max_age = max(ages) * 1.2
            age_range = np.linspace(0, max_age, 1000)
            biological_age_range = age_range * 365.0 * self._aging_rate
            completed_lifespan_ratio = np.minimum(1.0, biological_age_range / self._lifespan_in_days)
            completed_lifespan_ratio_with_mods = completed_lifespan_ratio * self._health_modifier
            non_linear_aging = np.array([BiologicalGrowthPatterns.gompertz_decay(x)
                                         for x in completed_lifespan_ratio_with_mods])
            theoretical_vitality = self._max_vitality * (1.0 - non_linear_aging)

            plt.plot(age_range, theoretical_vitality, '--', color='red', alpha=0.7,
                     label='Curva teórica')

        max_age_years = self._lifespan_in_days / 365.0
        plt.axvline(x=max_age_years, color='gray', linestyle=':',
                    label=f'Esperanza de vida ({max_age_years:.1f} años)')

        plt.title(f'Curva de Vitalidad (Tasa de envejecimiento: {self._aging_rate})')
        plt.xlabel('Edad (años)')
        plt.ylabel('Vitalidad')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        info_text = (
            f"Vitalidad máxima: {self._max_vitality}\n"
            f"Tasa de envejecimiento: {self._aging_rate}\n"
            f"Esperanza de vida: {max_age_years:.1f} años\n"
            f"Modificador de salud: {self._health_modifier}"
        )
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.tight_layout()
        plt.show()
