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

import numpy as np
from simpy import Environment as simpyEnv

from biome.components.physiological.heterotrophic_nutrition import HeterotrophicNutritionComponent
from biome.systems.components.managers.base import BaseComponentManager
from shared.enums.events import ComponentEvent
from shared.enums.reasons import StressReason
from shared.enums.strings import Loggers
from shared.enums.thresholds import ThirstThresholds, HungerThresholds
from shared.timers import Timers
from utils.loggers import LoggerManager


class HeterotrophicNutritionComponentManager(BaseComponentManager[HeterotrophicNutritionComponent]):
    def __init__(self, env: simpyEnv):
        super().__init__(env)

        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._env.process(self._update_all_metabolism(Timers.Components.Physiological.METABOLISM))

    def _update_all_metabolism(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                hunger_rates = np.array([comp.hunger_rate for comp in active_components])
                thirst_rates = np.array([comp.thirst_rate for comp in active_components])

                hunger_levels = np.array([comp.hunger_level for comp in active_components])
                thirst_levels = np.array([comp.thirst_level for comp in active_components])

                energy_levels = np.array([comp.energy_handler.energy_reserves for comp in active_components])
                max_energy = np.array([comp.energy_handler.max_energy_reserves for comp in active_components])

                new_hunger_levels = np.maximum(0.0, hunger_levels - hunger_rates)
                new_thirst_levels = np.maximum(0.0, thirst_levels - thirst_rates)

                critical_hunger_mask = new_hunger_levels < 100.0 * HungerThresholds.Level.CRITICAL
                low_hunger_mask = (new_hunger_levels >= 100.0 * HungerThresholds.Level.CRITICAL) & (
                        new_hunger_levels < 100.0 * HungerThresholds.Level.LOW)
                normal_hunger_mask = (new_hunger_levels >= 100.0 * HungerThresholds.Level.LOW) & (
                        new_hunger_levels < 100.0 * HungerThresholds.Level.NORMAL)
                satisfied_hunger_mask = new_hunger_levels >= 100.0 * HungerThresholds.Level.SATISFIED

                critical_thirst_mask = new_thirst_levels < 100.0 * ThirstThresholds.Level.CRITICAL
                low_thirst_mask = (new_thirst_levels >= 100.0 * ThirstThresholds.Level.CRITICAL) & (
                        new_thirst_levels < 100.0 * ThirstThresholds.Level.LOW)
                normal_thirst_mask = (new_thirst_levels >= 100.0 * ThirstThresholds.Level.LOW) & (
                        new_thirst_levels < 100.0 * ThirstThresholds.Level.NORMAL)
                satisfied_thirst_mask = new_thirst_levels >= 100.0 * ThirstThresholds.Level.SATISFIED

                energy_ratios = energy_levels / max_energy
                energy_stress_factors = np.zeros_like(energy_ratios)
                critical_energy_mask = energy_ratios < 0.1
                if np.any(critical_energy_mask):
                    energy_stress_factors[critical_energy_mask] = 1.0 * np.exp(
                        4.0 * (1.0 - energy_ratios[critical_energy_mask] / 0.1))

                energy_penalties = hunger_rates * 0.5 + thirst_rates * 0.5

                for i, component in enumerate(active_components):
                    old_hunger = component.hunger_level
                    component.hunger_level = new_hunger_levels[i]

                    if old_hunger != component.hunger_level:
                        component.event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            HeterotrophicNutritionComponent,
                            hunger_level=component.hunger_level
                        )

                    old_thirst = component.thirst_level
                    component.thirst_level = new_thirst_levels[i]

                    if old_thirst != component.thirst_level:
                        component.event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            HeterotrophicNutritionComponent,
                            thirst_level=component.thirst_level
                        )

                    if new_hunger_levels[i] <= 0.01 or new_thirst_levels[i] <= 0.01:
                        severe_penalty = component.energy_handler.max_energy_reserves * 0.05
                        energy_penalties[i] += severe_penalty
                        self._logger.debug(
                            f"CRITICAL condition! Hunger={new_hunger_levels[i]:.2f}, Thirst={new_thirst_levels[i]:.2f}. "
                            f"Applied severe energy penalty: -{severe_penalty:.2f}"
                        )

                    component.energy_handler.modify_energy(-energy_penalties[i])

                    if critical_hunger_mask[i]:
                        component.stress_handler.modify_stress(HungerThresholds.StressChange.CRITICAL,
                                                               StressReason.HUNGER)
                        self._logger.debug(f"CRITICAL hunger stress applied: +{HungerThresholds.StressChange.CRITICAL}")
                    elif low_hunger_mask[i]:
                        component.stress_handler.modify_stress(HungerThresholds.StressChange.LOW, StressReason.HUNGER)
                        self._logger.debug(f"LOW hunger stress applied: +{HungerThresholds.StressChange.LOW}")
                    elif normal_hunger_mask[i]:
                        component.stress_handler.modify_stress(HungerThresholds.StressChange.NORMAL,
                                                               StressReason.HUNGER)
                    elif satisfied_hunger_mask[i]:
                        component.stress_handler.modify_stress(HungerThresholds.StressChange.SATISFIED,
                                                               StressReason.HUNGER)
                        self._logger.debug(
                            f"SATISFIED hunger relief applied: {HungerThresholds.StressChange.SATISFIED}")

                    if critical_thirst_mask[i]:
                        component.stress_handler.modify_stress(ThirstThresholds.StressChange.CRITICAL,
                                                               StressReason.THIRST)
                        self._logger.debug(f"CRITICAL thirst stress applied: +{ThirstThresholds.StressChange.CRITICAL}")
                    elif low_thirst_mask[i]:
                        component.stress_handler.modify_stress(ThirstThresholds.StressChange.LOW, StressReason.THIRST)
                        self._logger.debug(f"LOW thirst stress applied: +{ThirstThresholds.StressChange.LOW}")
                    elif normal_thirst_mask[i]:
                        component.stress_handler.modify_stress(ThirstThresholds.StressChange.NORMAL,
                                                               StressReason.THIRST)
                    elif satisfied_thirst_mask[i]:
                        component.stress_handler.modify_stress(ThirstThresholds.StressChange.SATISFIED,
                                                               StressReason.THIRST)
                        self._logger.debug(
                            f"SATISFIED thirst relief applied: {ThirstThresholds.StressChange.SATISFIED}")

                    if energy_stress_factors[i] > 0.01:
                        component.stress_handler.modify_stress(energy_stress_factors[i], StressReason.NO_ENERGY)

                    self._logger.debug(
                        f"[Metabolic Update | {component.__class__.__name__}] "
                        f"Hunger: {component.hunger_level:.2f}, "
                        f"Thirst: {component.thirst_level:.2f}, "
                        f"Energy: {component.energy_reserves:.2f}/{component.max_energy_reserves:.2f}"
                    )

            yield self._env.timeout(timer)
