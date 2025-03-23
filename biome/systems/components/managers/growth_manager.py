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
from typing import Dict, List, Set, Optional
from simpy import Environment as simpyEnv

from biome.components.physiological.growth import GrowthComponent
from shared.enums.events import ComponentEvent
from shared.enums.strings import Loggers
from shared.math.biological import BiologicalGrowthPatterns
from shared.timers import Timers
from utils.loggers import LoggerManager


class GrowthComponentManager:
    def __init__(self, env: simpyEnv):
        self._env = env
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._components: Dict[int, GrowthComponent] = {}
        self._component_ids = set()
        self._process = self._env.process(self._update_all_growth(Timers.Compoments.Physiological.GROWTH))

    def register_component(self, component_id: int, component: GrowthComponent) -> None:
        self._components[component_id] = component
        self._component_ids.add(component_id)

    def unregister_component(self, component_id: int) -> None:
        if component_id in self._components:
            del self._components[component_id]
            self._component_ids.discard(component_id)

    def _get_active_components(self) -> List[GrowthComponent]:
        return [comp for comp in self._components.values()
                if comp._host_alive and not comp._is_dormant]

    def _update_all_growth(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                current_sizes = []
                max_sizes = []
                growth_modifiers = []
                growth_efficiencies = []
                ages = []
                lifespans = []
                initial_sizes = []

                for comp in active_components:
                    current_sizes.append(comp._current_size)
                    max_sizes.append(comp._max_size)
                    growth_modifiers.append(comp._growth_modifier)
                    growth_efficiencies.append(comp._growth_efficiency)
                    ages.append(comp._env.now)
                    lifespans.append(comp._lifespan_in_ticks)
                    initial_sizes.append(comp._initial_size)

                current_sizes = np.array(current_sizes)
                max_sizes = np.array(max_sizes)
                growth_modifiers = np.array(growth_modifiers)
                growth_efficiencies = np.array(growth_efficiencies)
                ages = np.array(ages)
                lifespans = np.array(lifespans)
                initial_sizes = np.array(initial_sizes)

                biological_age_ratios = ages / lifespans
                biological_age_ratios_with_mods = biological_age_ratios * growth_modifiers * growth_efficiencies
                biological_age_ratios_with_mods = np.minimum(1.0, biological_age_ratios_with_mods)

                non_linear_growth_completion_ratios = BiologicalGrowthPatterns.sigmoid_growth_curve_vectorized(
                    biological_age_ratios_with_mods, 6)

                new_sizes = initial_sizes + (max_sizes - initial_sizes) * non_linear_growth_completion_ratios

                for i, component in enumerate(active_components):
                    if component._current_size < component._max_size:
                        old_size = component._current_size
                        component._current_size = new_sizes[i]

                        component._event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            GrowthComponent,
                            current_size=component._current_size,
                            tick=component._env.now
                        )

                        for stage, threshold in enumerate(component._stage_thresholds):
                            if component._growth_stage == stage and component._current_size >= threshold:
                                component._growth_stage += 1
                                self._logger.warning(f"Increasing stage: {component._growth_stage}")
                                component._event_notifier.notify(
                                    ComponentEvent.UPDATE_STATE,
                                    GrowthComponent,
                                    growth_stage=component._growth_stage
                                )
                                component._event_notifier.notify(
                                    ComponentEvent.STAGE_CHANGE,
                                    stage=stage
                                )
                                break

            yield self._env.timeout(timer)