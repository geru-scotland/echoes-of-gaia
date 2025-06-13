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
Growth lifecycle management for biome entity development.

Controls entity growth progression through sigmoid growth patterns;
updates size attributes based on age, efficiency, and environmental factors.
Manages growth stage transitions and notifies relevant systems - supports
non-linear biological development with realistic progression curves.
"""

from logging import Logger

import numpy as np
from typing import Dict, List, Set, Optional
from simpy import Environment as simpyEnv

from biome.components.physiological.growth import GrowthComponent
from biome.systems.components.managers.base import BaseComponentManager
from shared.enums.events import ComponentEvent
from shared.enums.strings import Loggers
from shared.math.biological import BiologicalGrowthPatterns
from shared.timers import Timers
from utils.loggers import LoggerManager


class GrowthComponentManager(BaseComponentManager[GrowthComponent]):
    def __init__(self, env: simpyEnv):
        super().__init__(env)
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._process = self._env.process(self._update_all_growth(Timers.Components.Physiological.GROWTH))

    def _update_all_growth(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()
            self._logger.debug(f"Número de componentes de crecimiento activos: {len(active_components)}")

            if active_components:

                max_sizes = np.array([comp.max_size for comp in active_components])
                growth_modifiers = np.array([comp.growth_modifier for comp in active_components])
                growth_efficiencies = np.array([comp.growth_efficiency for comp in active_components])
                ages = np.array([self._env.now for _ in active_components])
                lifespans = np.array([comp.lifespan for comp in active_components])
                initial_sizes = np.array([comp.initial_size for comp in active_components])

                biological_age_ratios = ages / lifespans
                biological_age_ratios_with_mods = biological_age_ratios * growth_modifiers * growth_efficiencies
                biological_age_ratios_with_mods = np.minimum(1.0, biological_age_ratios_with_mods)

                non_linear_growth_completion_ratios = BiologicalGrowthPatterns.sigmoid_growth_curve_vectorized(
                    biological_age_ratios_with_mods, 6)

                new_sizes = initial_sizes + (max_sizes - initial_sizes) * non_linear_growth_completion_ratios

                for i, component in enumerate(active_components):
                    if component.current_size < component.max_size:
                        old_size = component.current_size
                        component.current_size = new_sizes[i]

                        self._logger.debug(f"Actualizando tamaño de {old_size} a {component.current_size}")

                        component.event_notifier.notify(
                            ComponentEvent.UPDATE_STATE,
                            GrowthComponent,
                            current_size=component.current_size,
                            tick=self._env.now
                        )

                        for stage, threshold in enumerate(component.stage_thresholds):
                            if component.growth_stage == stage and component.current_size >= threshold:
                                component.growth_stage += 1

                                component.event_notifier.notify(
                                    ComponentEvent.UPDATE_STATE,
                                    GrowthComponent,
                                    growth_stage=component.growth_stage
                                )
                                component.event_notifier.notify(
                                    ComponentEvent.STAGE_CHANGE,
                                    stage=stage
                                )
                                break

            yield self._env.timeout(timer)
