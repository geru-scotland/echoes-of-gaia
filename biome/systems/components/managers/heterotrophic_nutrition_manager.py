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

                hunger_stress_factors = (1.0 - new_hunger_levels / 100.0) * 0.1
                thirst_stress_factors = (1.0 - new_thirst_levels / 100.0) * 0.15

                critical_hunger_mask = new_hunger_levels < 10.0

                if np.any(critical_hunger_mask):
                    critical_factors = 0.5 * np.exp(5.0 * (1.0 - new_hunger_levels[critical_hunger_mask] / 10.0))
                    hunger_stress_factors[critical_hunger_mask] = critical_factors

                critical_thirst_mask = new_thirst_levels < 10.0
                if np.any(critical_thirst_mask):
                    critical_factors = 0.8 * np.exp(6.0 * (1.0 - new_thirst_levels[critical_thirst_mask] / 10.0))
                    thirst_stress_factors[critical_thirst_mask] = critical_factors

                energy_ratios = energy_levels / max_energy
                energy_stress_factors = np.zeros_like(energy_ratios)
                critical_energy_mask = energy_ratios < 0.1
                if np.any(critical_energy_mask):
                    energy_stress_factors[critical_energy_mask] = 1.0 * np.exp(
                        4.0 * (1.0 - energy_ratios[critical_energy_mask] / 0.1))

                multi_critical_factors = np.zeros_like(hunger_stress_factors)
                multi_critical_mask = (new_hunger_levels < 15.0) & (new_thirst_levels < 15.0)
                if np.any(multi_critical_mask):
                    multi_critical_factors[
                        multi_critical_mask] = 2.0

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

                    component.energy_handler.modify_energy(-energy_penalties[i])

                    if hunger_stress_factors[i] > 0.01:
                        component.stress_handler.modify_stress(hunger_stress_factors[i], StressReason.HUNGER)

                    if thirst_stress_factors[i] > 0.01:
                        component.stress_handler.modify_stress(thirst_stress_factors[i], StressReason.THIRST)

                    if energy_stress_factors[i] > 0.01:
                        component.stress_handler.modify_stress(energy_stress_factors[i], StressReason.NO_ENERGY)

                    if multi_critical_factors[i] > 0:
                        component.stress_handler.modify_stress(multi_critical_factors[i],
                                                               StressReason.CRITICAL_CONDITION)

                    self._logger.debug(
                        f"[Metabolic Update | {component.__class__.__name__}] "
                        f"Hunger: {component.hunger_level:.2f}, "
                        f"Thirst: {component.thirst_level:.2f}, "
                        f"Energy: {component.energy_reserves:.2f}/{component.max_energy_reserves:.2f}"
                    )

            yield self._env.timeout(timer)
