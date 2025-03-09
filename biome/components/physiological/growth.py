"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from typing import Optional, List

import math
from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.services.climate_service import ClimateService
from biome.systems.climate.state import ClimateState
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent, BiomeEvent
from shared.timers import Timers

class BiologicalGrowthPatterns:

    @staticmethod
    def sigmoid_growth_curve(relative_age: float, curve_steepness: float = 10) -> float:
        min_bound = 1 / (1 + math.exp(curve_steepness / 2))
        max_bound = 1 / (1 + math.exp(-curve_steepness / 2))

        raw_growth_value = 1 / (1 + math.exp(-curve_steepness * (relative_age - 0.5)))

        return (raw_growth_value - min_bound) / (max_bound - min_bound)

    @staticmethod
    def von_bertalanffy_growth(relative_age: float, growth_constant: float = 2.0) -> float:
        return 1.0 - math.exp(-growth_constant * relative_age)

class GrowthComponent(EntityComponent):
    def __init__(self, env: simpyEnv,
                 event_notifier: EventNotifier,
                 lifespan: float = 15.0,
                 growth_stage: int = 0,
                 total_stages: int = 4,
                 current_size: float = 0.05,
                 max_size: float = 3.0,
                 growth_modifier: float = 1.0,
                 growth_efficiency: float = 0.85):
        super().__init__(env, ComponentType.GROWTH, event_notifier)

        self._lifespan_in_ticks: float = lifespan * float(Timers.Calendar.YEAR)
        self._growth_stage: int = growth_stage
        self._initial_size = current_size
        self._total_stages: int = total_stages
        self._current_size: float = round(current_size, 3)
        self._max_size: float = round(max_size, 3)

        self._growth_modifier: float = growth_modifier # Este lo cambio SOLO por evolución
        # Y este, es el que modifico por estrés, clima,
        # enfermedades... es cómo de bien convierte recursos
        self._growth_efficiency: float = round(growth_efficiency, 3)

        self._stage_thresholds: List[float] = self._calculate_stage_thresholds()

        self._log_data("Initialized")

        self._env.process(self._update_growth(Timers.Compoments.Physiological.GROWTH))

    def _register_events(self):
        self._event_notifier.register(ComponentEvent.COLD_WEATHER, self._handle_cold_weather)

    def _calculate_stage_thresholds(self) -> List[float]:
        return [self._max_size * (i / self._total_stages) for i in range(1, self._total_stages + 1)]

    def _update_growth_modifier(self, modifier: float) -> None:
        self._growth_modifier: float = max(0.0, modifier)
        self._event_notifier.notify(ComponentEvent.UPDATE_STATE, GrowthComponent,
                                    growth_modifier=self._growth_modifier)

    def _update_growth_efficiency(self, modifier: float) -> None:
        self._growth_efficiency: float = max(0.0, modifier)
        self._event_notifier.notify(ComponentEvent.UPDATE_STATE, GrowthComponent,
                                    growth_efficiency=self._growth_efficiency)

    def _handle_cold_weather(self, *args, **kwargs):
        if self._growth_stage == self._total_stages:
            return
        # TODO: Guardar histórico de temperaturas EN EL CLIMATE SYSTEM
        # No aquí, ni de coña, y que lo agregue al ClimateState
        state: ClimateState = ClimateService.query_state()
        # self._update_growth_efficiency(efficiency)


    def _update_growth(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            if self._current_size < self._max_size:

                biological_age_ratio: float = self._env.now / self._lifespan_in_ticks
                biological_age_ratio_with_mods =  biological_age_ratio * self._growth_modifier * self._growth_efficiency
                biological_age_ratio_with_mods = min(1.0, biological_age_ratio_with_mods)

                # Quiero transformar el tiempo lineal, que aumenta constantemente, en un patrón
                # de crecimiento biológico NO lienal (ya sabes, S)
                non_lineal_growth_completion_ratio = BiologicalGrowthPatterns.sigmoid_growth_curve(biological_age_ratio_with_mods)
                self._current_size = self._initial_size + (self._max_size - self._initial_size) * non_lineal_growth_completion_ratio

                self._event_notifier.notify(ComponentEvent.UPDATE_STATE, GrowthComponent,
                                            current_size=self._current_size, tick=self._env.now)

                self._log_data(f"Updating")
                for stage, threshold in enumerate(self._stage_thresholds):
                    if self._growth_stage == stage and self._current_size >= threshold:
                        self._growth_stage += 1
                        self._event_notifier.notify(ComponentEvent.UPDATE_STATE, GrowthComponent,
                                                    growth_stage=self._growth_stage)
                        self._event_notifier.notify(ComponentEvent.STAGE_CHANGE, stage=stage)
                        self._logger.debug(f"Advanced to growth stage {self._growth_stage}")
                        break

            yield self._env.timeout(timer)

    def _log_data(self, message: Optional[str]):
        if message:
            self._logger.debug(message)
        state: ClimateState = ClimateService.query_state()
        if state:
            self._logger.info(f"Temperature: {state.temperature}")
        self._logger.info(f" [Tick: {self._env.now} Growth: Stage={self._growth_stage}/{self._total_stages}, "
                      f"Size={self._current_size}/{self._max_size}, "
                      f"Efficiency={self._growth_efficiency}")

    @property
    def current_size(self) -> float:
        return self._current_size

    @property
    def growth_stage(self) -> int:
        return self._growth_stage

    @property
    def max_size(self) -> float:
        return self._max_size

    @property
    def is_mature(self) -> bool:
        return self._growth_stage == self._total_stages - 1