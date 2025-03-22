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
import random
from typing import Optional, Dict, Any

from biome.components.base.component import EntityComponent
from simpy import Environment as simpyEnv

from biome.components.handlers.energy_handler import EnergyHandler
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.enums.thresholds import MetabolicThresholds
from shared.timers import Timers


class PhotosyntheticMetabolismComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 photosynthesis_efficiency: float = 0.75, respiration_rate: float = 0.05,
                 metabolic_activity: float = 1.0, max_energy_reserves: float = 100.0,
                 optimal_respiration_ratio: float = 0.2):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.PHOTOSYNTHETIC_METABOLISM, event_notifier, lifespan)

        self._base_photosynthesis_efficiency: float = round(photosynthesis_efficiency, 2)
        self._photosynthesis_efficiency: float = self._base_photosynthesis_efficiency
        self._base_respiration_rate: float = round(respiration_rate, 4)
        self._respiration_rate: float = self._base_respiration_rate
        self._metabolic_activity: float = round(metabolic_activity, 2)
        # TODO: Para light availability, hace falta más información en el state
        # quizá agregar el weather effect y analizar desde aquí, mediante ClimateService
        self._light_availability: float = 0.7
        self._temperature_modifier: float = 1.0
        self._water_modifier: float = 0.7
        self._biological_age: Optional[float] = None
        self._optimal_respiration_ratio: float = optimal_respiration_ratio
        self._energy_factor: float = self._energy_handler.energy_reserves / self._energy_handler.max_energy_reserves


        self._logger.debug(f"Metabolic component initialized: Photosynthesis={self._photosynthesis_efficiency}, "
                           f"Respiration={self._respiration_rate}, Energy={self._energy_handler.energy_reserves}")

        self._env.process(self._update_metabolic_stress(Timers.Compoments.Physiological.STRESS_UPDATE))
        self._env.process(self._update_metabolism(Timers.Compoments.Physiological.METABOLISM))

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._energy_handler.register_events()
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)
        self._event_notifier.register(ComponentEvent.BIOLOGICAL_AGE_UPDATED, self._handle_biological_age_update)

    def _handle_biological_age_update(self, *args, **kwargs) -> None:
        self._biological_age = kwargs.get("biological_age", None)

    def _handle_energy_update(self, *args, **kwargs) -> None:
        energy_reserves: float = kwargs.get("energy_reserves", 0.0)
        self._energy_factor = energy_reserves / self._energy_handler.max_energy_reserves

    def _update_metabolic_stress(self, timer: int):
        yield self._env.timeout(timer)

        while self._host_alive:

            if self._energy_factor <= 0.0005:
                stress_change = MetabolicThresholds.StressChange.NO_ENERGY
                self._stress_handler.modify_stress(stress_change, StressReason.NO_ENERGY)

            elif MetabolicThresholds.Energy.CRITICAL > self._energy_factor > 0.0:
                stress_change = MetabolicThresholds.StressChange.CRITICAL
                self._stress_handler.modify_stress(stress_change, StressReason.NUTRIENT_DEFICIENCY)

            elif self._energy_factor < MetabolicThresholds.Energy.LOW:
                stress_change = MetabolicThresholds.StressChange.LOW
                self._stress_handler.modify_stress(stress_change, StressReason.NUTRIENT_DEFICIENCY)

            elif self._energy_factor > MetabolicThresholds.Energy.ABUNDANT:
                stress_change = MetabolicThresholds.StressChange.ABUNDANT
                self._stress_handler.modify_stress(stress_change, StressReason.ENERGY_ABUNDANCE)

            elif self._energy_factor > MetabolicThresholds.Energy.SUFFICIENT:
                stress_change = MetabolicThresholds.StressChange.SUFFICIENT
                self._stress_handler.modify_stress(stress_change, StressReason.ENERGY_SUFFICIENT)

            yield self._env.timeout(timer)

    def _update_metabolism(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while self._host_alive:
            effective_photosynthesis = (
                    self._photosynthesis_efficiency *
                    self._light_availability *
                    self._temperature_modifier *
                    self._water_modifier *
                    self._metabolic_activity *
                    (0.15 if self._is_dormant else 1.0)
            )
            effective_respiration = (
                    self._respiration_rate *
                    self._metabolic_activity *
                    (0.1 if self._is_dormant else 1.0)
            )

            current_ratio = effective_respiration / max(0.001, effective_photosynthesis)
            ratio_difference = abs(current_ratio - self._optimal_respiration_ratio)

            metabolic_efficiency = max(0.1, 1.0 - (ratio_difference / self._optimal_respiration_ratio) * 1.5)

            base_energy_change = (effective_photosynthesis - effective_respiration)
            base_energy_change *= metabolic_efficiency

            entropy_decay = 0.02
            age_factor = 1.0
            age_decay = 0.0

            if self._biological_age and self._lifespan:
                relative_age = self._biological_age / (self._lifespan * float(Timers.Calendar.YEAR))

                age_factor = 1.0 + (relative_age ** 2) * 0.5

                age_decay = max(0.01, self._energy_handler.energy_reserves * 0.002 * age_factor)

            final_energy_change = base_energy_change - (entropy_decay + age_decay)

            self._energy_handler.modify_energy(final_energy_change)

            self._logger.debug(
                f"[Metabolic Update | DEBUG:Tick={self._env.now}]"
                f"Stress level={self._stress_handler.stress_level:4f}]"
                f"Energy: {self._energy_handler.energy_reserves:.3f}/{self._energy_handler.max_energy_reserves:.3f}, "
                f"Photosynthesis: Eff={self._photosynthesis_efficiency:.3f}, "
                f"Light={self._light_availability:.3f}, "
                f"Temp Mod={self._temperature_modifier:.3f}, "
                f"Water Mod={self._water_modifier:.3f}, "
                f"Activity={self._metabolic_activity:.3f}, "
                f"Respiration: Rate={self._respiration_rate:.3f}, "
                f"Energy Change={final_energy_change:.3f}, "
                f"Dormant={self._is_dormant}"
            )
            yield self._env.timeout(timer)

    def _handle_stress_update(self, *args, **kwargs):

        normalized_stress: float = kwargs.get("normalized_stress", 0.0)
        old_efficiency: float = self._photosynthesis_efficiency
        new_efficiency = self._base_photosynthesis_efficiency

        if 0.0 <= normalized_stress < 0.4:
            adaptation_factor = 1.0 + 0.25 * (1.0 - (normalized_stress - 0.25) ** 2 * 16)
            adapted_efficiency = min(self._base_photosynthesis_efficiency * 1.5,
                                     self._base_photosynthesis_efficiency * adaptation_factor)

            delta_adaptation = adapted_efficiency - self._base_photosynthesis_efficiency
            new_efficiency += delta_adaptation

        stress_impact_photosynthesis = math.exp(-1.5 * normalized_stress)
        non_adapted_efficiency = max(
            0.25,
            self._base_photosynthesis_efficiency * stress_impact_photosynthesis
        )

        delta_non_adapted = non_adapted_efficiency - self._base_photosynthesis_efficiency
        new_efficiency += delta_non_adapted

        self._photosynthesis_efficiency = max(
            0.25,
            min(self._base_photosynthesis_efficiency * 1.5, new_efficiency)
        )

        stress_impact_respiration = 1 + (1.0 * normalized_stress ** 2)
        self._respiration_rate = min(
            0.2,
            self._base_respiration_rate * stress_impact_respiration
        )

        self._event_notifier.notify(
            ComponentEvent.PHOTOSYNTHESIS_UPDATED,
            old_efficiency=old_efficiency,
            new_efficiency=self._photosynthesis_efficiency,
        )

        self._logger.debug(
            f"Stress Update Received: Norm.Stress={normalized_stress:.2f}, "
            f"Photosynthesis={self._photosynthesis_efficiency:.2f}, "
            f"Respiration={self._respiration_rate:.2f}"
        )

    def set_environmental_modifiers(self, light: float = None, temperature: float = None, water: float = None):
        if light is not None:
            self._light_availability = max(0.0, min(1.0, light))

        if temperature is not None:
            self._temperature_modifier = max(0.0, min(1.5, temperature))

        if water is not None:
            self._water_modifier = max(0.0, min(1.0, water))

    def set_metabolic_activity(self, activity_level: float):
        self._metabolic_activity = max(0.1, min(1.0, activity_level))

    def consume_energy(self, amount: float) -> bool:
        if amount <= self._energy_handler.energy_reserves:
            new_energy: float = self._energy_handler.energy_reserves - amount
            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, PhotosyntheticMetabolismComponent,
                                        energy_reserves=new_energy)
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "energy_reserves": self._energy_handler.energy_reserves,
            "max_energy_reserves": self._energy_handler.max_energy_reserves,
            "photosynthesis_efficiency": self._photosynthesis_efficiency,
            "respiration_rate": self._respiration_rate,
            "metabolic_activity": self._metabolic_activity,
        }

    @property
    def base_photosynthesis_efficiency(self) -> float:
        return self._base_photosynthesis_efficiency

    @property
    def base_respiration_rate(self) -> float:
        return self._base_respiration_rate

    @property
    def metabolic_activity(self) -> float:
        return self._metabolic_activity

    @property
    def energy_reserves(self) -> float:
        return self._energy_handler.energy_reserves

    @property
    def max_energy_reserves(self) -> float:
        return self._energy_handler.max_energy_reserves