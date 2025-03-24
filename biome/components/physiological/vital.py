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
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from simpy import Environment as simpyEnv
from biome.components.base.component import EntityComponent
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.components.registry import ComponentRegistry
from shared.math.biological import BiologicalGrowthPatterns
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.enums.thresholds import VitalThresholds
from shared.timers import Timers
from simulation.core.systems.events.event_bus import SimulationEventBus


class VitalComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float = 5.0, health_modifier: float = 1.0,
                 vitality: float = 100.0, max_vitality: float = 100.0, age: float = 0.0, aging_rate: float = 1.0,
                 dormancy_threshold: float = 25.0):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)

        super().__init__(env, ComponentType.VITAL, event_notifier, lifespan)

        self._lifespan_in_ticks: int = int((lifespan * float(Timers.Calendar.YEAR)))
        self._vitality: float = round(vitality, 2)
        self._max_vitality: float = round(max_vitality, 2)
        self._age: float = age
        self._aging_rate: float = aging_rate
        self._biological_age: float = self._age * aging_rate
        self._dormancy_threshold: float = round(dormancy_threshold, 2)
        self._health_modifier: float = 1.0
        self._vitality_history: List[Tuple[float, float]] = []  # (age, vitality)
        self._birth_tick: int = self._env.now

        self._logger.debug(f"Vital component initialized: Health={self._vitality}/{self._max_vitality}, "
                           f"Age={self._age}")

        ComponentRegistry.get_vital_manager().register_component(id(self), self)

    def _register_events(self) -> None:
        super()._register_events()
        self._stress_handler.register_events()
        SimulationEventBus.register("simulation_finished", self._handle_simulation_finished)

    def _handle_simulation_finished(self) -> None:
        # self.plot_vitality_curve()
        pass

    def disable_notifier(self):
        super().disable_notifier()
        ComponentRegistry.get_vital_manager().unregister_component(id(self))

    def get_state(self) -> Dict[str, Any]:
        return {
            "birth_tick": self._birth_tick,
            "age": self._age,
            "biological_age": self._biological_age,
            "aging_rate": self._aging_rate,
            "vitality": self._vitality,
            "max_vitality": self._max_vitality,
            "health_modifier": self._health_modifier,
        }

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
            biological_age_range = age_range * float(Timers.Calendar.YEAR) * self._aging_rate
            completed_lifespan_ratio = np.minimum(1.0, biological_age_range / self._lifespan_in_ticks)
            completed_lifespan_ratio_with_mods = completed_lifespan_ratio * self._health_modifier
            non_linear_aging = np.array([BiologicalGrowthPatterns.gompertz_decay(x)
                                         for x in completed_lifespan_ratio_with_mods])
            theoretical_vitality = self._max_vitality * (1.0 - non_linear_aging)

            plt.plot(age_range, theoretical_vitality, '--', color='red', alpha=0.7,
                     label='Curva teórica')

        max_age_years = self._lifespan_in_ticks / 720.0
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

    @property
    def max_vitality(self) -> float:
        return self._max_vitality

    @property
    def aging_rate(self) -> float:
        return self._aging_rate

    @property
    def health_modifier(self) -> float:
        return self._health_modifier

    @property
    def lifespan(self) -> float:
        return self._lifespan_in_ticks / (float(Timers.Calendar.YEAR) / float(Timers.Calendar.DAY))

    @property
    def stress_level(self) -> float:
        return self._stress_handler.stress_level