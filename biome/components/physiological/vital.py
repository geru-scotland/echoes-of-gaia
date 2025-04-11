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
from biome.components.handlers.energy_handler import EnergyHandler
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.components.registry import ComponentRegistry
from shared.enums.events import SimulationEvent, ComponentEvent
from shared.math.biological import BiologicalGrowthPatterns
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.timers import Timers
from simulation.core.systems.events.event_bus import SimulationEventBus


class VitalComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float = 5.0,
                 health_modifier: float = 1.0,
                 vitality: float = 100.0, max_vitality: float = 100.0, age: float = 0.0, aging_rate: float = 1.0,
                 dormancy_threshold: float = 25.0, max_energy_reserves: float = 100.0):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.VITAL, event_notifier, lifespan)

        self._lifespan_in_ticks: int = int((lifespan * float(Timers.Calendar.YEAR)))
        self._vitality: float = round(max_vitality, 2)
        self._max_vitality: float = round(max_vitality, 2)
        self._age: float = age
        self._aging_rate: float = aging_rate
        self._biological_age: float = self._age * aging_rate
        self._dormancy_threshold: float = round(dormancy_threshold, 2)
        self._health_modifier: float = 1.0
        self._vitality_history: List[Tuple[float, float]] = []  # (age, vitality)
        self._birth_tick: int = self._env.now
        self._accumulated_decay: float = 0.0
        self._accumulated_stress: float = 0.0
        self._logger.debug(f"Vital component initialized: Health={self._vitality}/{self._max_vitality}, "
                           f"Age={self._age}")

        self._somatic_integrity: float = 100.0
        self._max_somatic_integrity: float = 100.0
        self._integrity_regeneration_rate: float = 0.5
        ComponentRegistry.get_vital_manager().register_component(id(self), self)

    def _register_events(self) -> None:
        super()._register_events()
        self._stress_handler.register_events()
        self._energy_handler.register_events()
        SimulationEventBus.register(SimulationEvent.SIMULATION_FINISHED, self._handle_simulation_finished)

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
            "somatic_integrity": self._somatic_integrity,
            "max_somatic_integrity": self._max_somatic_integrity
        }

    def increase_accumulated_decay(self, value: float) -> None:
        self._accumulated_decay += value

    def increase_accumulated_stress(self, value: float) -> None:
        self._accumulated_stress += value

    def apply_damage(self, damage_amount: float, source_entity_id: Optional[int] = None) -> None:
        if not self._host_alive:
            return

        old_integrity = self._somatic_integrity
        self._somatic_integrity = max(0, self._somatic_integrity - damage_amount)
        self._logger.debug(f"APPLYING DAMAGE. Damage: {damage_amount} Somatic integrity: {self.somatic_integrity}")

        self._event_notifier.notify(
            ComponentEvent.UPDATE_STATE,
            VitalComponent,
            somatic_integrity=self._somatic_integrity
        )

        self._logger.debug(
            f"Physical damage applied: -{damage_amount:.2f} (from {old_integrity:.2f} to {self._somatic_integrity:.2f})")

        if self._somatic_integrity <= 0:
            self._vitality = 0
            self._event_notifier.notify(
                ComponentEvent.ENTITY_DEATH,
                component=ComponentType.VITAL,
                cleanup_dead_entities=True
            )

    def heal_integrity(self, healing_amount: float) -> None:

        if not self._host_alive:
            return

        old_integrity = self._somatic_integrity
        self._somatic_integrity = min(self._max_somatic_integrity, self._somatic_integrity + healing_amount)

        self._event_notifier.notify(
            ComponentEvent.UPDATE_STATE,
            VitalComponent,
            somatic_integrity=self._somatic_integrity
        )

        self._logger.debug(
            f"Physical integrity healed: +{healing_amount:.2f} (from {old_integrity:.2f} to {self._somatic_integrity:.2f})")

    def get_integrity_percentage(self) -> float:
        return (self._somatic_integrity / self._max_somatic_integrity) * 100.0

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

    # Getters
    @property
    def vitality(self) -> float:
        return self._vitality

    @property
    def max_vitality(self) -> float:
        return self._max_vitality

    @property
    def age(self) -> float:
        return self._age

    @property
    def biological_age(self) -> float:
        return self._biological_age

    @property
    def aging_rate(self) -> float:
        return self._aging_rate

    @property
    def birth_tick(self) -> int:
        return self._birth_tick

    @property
    def dormancy_threshold(self) -> float:
        return self._dormancy_threshold

    @property
    def health_modifier(self) -> float:
        return self._health_modifier

    @property
    def vitality_history(self) -> List[Tuple[float, float]]:
        return self._vitality_history

    @property
    def lifespan_in_ticks(self) -> int:
        return self._lifespan_in_ticks

    @property
    def lifespan(self) -> int:
        return self._lifespan_in_ticks

    @property
    def stress_level(self) -> float:
        return self._stress_handler.stress_level

    @property
    def stress_handler(self) -> StressHandler:
        return self._stress_handler

    @property
    def event_notifier(self) -> EventNotifier:
        return self._event_notifier

    @property
    def is_active(self) -> bool:
        return self._host_alive

    @property
    def energy_handler(self) -> EnergyHandler:
        return self._energy_handler

    @property
    def energy_reserves(self) -> float:
        return self._energy_handler.energy_reserves

    @property
    def accumulated_decay(self) -> float:
        return self._accumulated_decay

    @property
    def accumulated_stress(self) -> float:
        return self._accumulated_stress

    @property
    def somatic_integrity(self) -> float:
        return self._somatic_integrity

    @property
    def max_somatic_integrity(self) -> float:
        return self._max_somatic_integrity

    @property
    def somatic_integrity_regen_rate(self) -> float:
        return self._integrity_regeneration_rate

        # Setters

    @vitality.setter
    def vitality(self, value: float) -> None:
        self._vitality = value

    @max_vitality.setter
    def max_vitality(self, value: float) -> None:
        self._max_vitality = value

    @age.setter
    def age(self, value: float) -> None:
        self._age = value

    @biological_age.setter
    def biological_age(self, value: float) -> None:
        self._biological_age = value

    @aging_rate.setter
    def aging_rate(self, value: float) -> None:
        self._aging_rate = value

    @birth_tick.setter
    def birth_tick(self, value: int) -> None:
        self._birth_tick = value

    @dormancy_threshold.setter
    def dormancy_threshold(self, value: float) -> None:
        self._dormancy_threshold = value

    @health_modifier.setter
    def health_modifier(self, value: float) -> None:
        self._health_modifier = value

    @vitality_history.setter
    def vitality_history(self, value: List[Tuple[float, float]]) -> None:
        self._vitality_history = value

    @lifespan_in_ticks.setter
    def lifespan_in_ticks(self, value: int) -> None:
        self._lifespan_in_ticks = value

    @lifespan.setter
    def lifespan(self, value: int) -> None:
        self._lifespan = value
