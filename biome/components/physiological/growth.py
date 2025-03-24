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
from typing import Optional, List, Dict, Any

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.timers import Timers


class GrowthComponent(EntityComponent):

    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float = 15.0,
                 growth_stage: int = 0, total_stages: int = 4, current_size: float = 0.05, max_size: float = 3.0,
                 growth_modifier: float = 1.0, growth_efficiency: float = 0.85):

        super().__init__(env, ComponentType.GROWTH, event_notifier, lifespan)

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

        # En lugar de crear su propio proceso SimPy, lo registro en el manager
        ComponentRegistry.get_growth_manager().register_component(id(self), self)

    def _register_events(self):
        super()._register_events()

    def _calculate_stage_thresholds(self) -> List[float]:
        total_stages_int = int(self._total_stages)
        return [self._max_size * (i / total_stages_int) for i in range(1, total_stages_int + 1)]

    def _handle_stress_update(self, *args, **kwargs):
        super()._handle_stress_update(*args, **kwargs)

        normalized_stress = kwargs.get("normalized_stress", 0.0)
        # TODO: pasar todos los factores a config, especificos para estrés, 0.6 etc.
        self._growth_efficiency = max(0.3, self._growth_efficiency * (1.0 - normalized_stress * 0.6))

    def disable_notifier(self):
        super().disable_notifier()
        ComponentRegistry.get_growth_manager().unregister_component(id(self))

    def _log_data(self, message: Optional[str]):
        if message:
            self._logger.debug(message)

        self._logger.debug(f" [Tick: {self._env.now} Growth: Stage={self._growth_stage}/{self._total_stages}, "
                      f"Size={self._current_size}/{self._max_size}, "
                      f"Efficiency={self._growth_efficiency}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "growth_stage": self._growth_stage,
            "current_size": self._current_size,
            "max_size": self._max_size,
            "growth_efficiency": self._growth_efficiency
        }

    # Getters
    @property
    def growth_stage(self) -> int:
        return self._growth_stage

    @property
    def current_size(self) -> float:
        return self._current_size

    @property
    def max_size(self) -> float:
        return self._max_size

    @property
    def growth_modifier(self) -> float:
        return self._growth_modifier

    @property
    def growth_efficiency(self) -> float:
        return self._growth_efficiency

    @property
    def initial_size(self) -> float:
        return self._initial_size

    @property
    def total_stages(self) -> int:
        return self._total_stages

    @property
    def stage_thresholds(self) -> List[float]:
        return self._stage_thresholds

    @property
    def lifespan(self) -> float:
        return self._lifespan_in_ticks

    @property
    def event_notifier(self) -> EventNotifier:
        return self._event_notifier

    @property
    def is_active(self) -> bool:
        return self._host_alive

    # Setters
    @growth_stage.setter
    def growth_stage(self, value: int) -> None:
        self._growth_stage = value

    @current_size.setter
    def current_size(self, value: float) -> None:
        self._current_size = value

    @max_size.setter
    def max_size(self, value: float) -> None:
        self._max_size = value

    @growth_modifier.setter
    def growth_modifier(self, value: float) -> None:
        self._growth_modifier = value

    @growth_efficiency.setter
    def growth_efficiency(self, value: float) -> None:
        self._growth_efficiency = value

    @initial_size.setter
    def initial_size(self, value: float) -> None:
        self._initial_size = value
