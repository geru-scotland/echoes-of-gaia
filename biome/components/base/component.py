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
from logging import Logger
from abc import abstractmethod, ABC
from typing import Optional, Set

from simpy import Environment as simpyEnv

from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.enums.strings import Loggers
from shared.enums.thresholds import ClimateThresholds
from shared.events.handler import EventHandler
from utils.loggers import LoggerManager


class Component(ABC):
    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._type: ComponentType = type
        self._env: simpyEnv = env
        self._event_notifier: EventNotifier = event_notifier

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def _update(self, delay: Optional[int] = None):
        pass

    @property
    def type(self):
        return self._type


class BiomeComponent(Component):
    def __init__(self, type: ComponentType, env: simpyEnv):
        super().__init__(type, env)

    def get_state(self):
        pass

    def _update(self, delay: Optional[int] = None):
        pass


class EntityComponent(Component, EventHandler):
    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier):
        Component.__init__(self, env, type, event_notifier)
        EventHandler.__init__(self)

    @abstractmethod
    def _register_events(self):
       raise NotImplementedError

    def get_state(self):
        pass

    def _update(self, delay: Optional[int] = None):
        pass

class FloraComponent(EntityComponent):

    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier):
        super().__init__(env, type, event_notifier)
        self._event_notifier: EventNotifier = event_notifier
        self._dormancy_reasons: Set[DormancyReason] = set()
        self._is_dormant: bool = False
        self._stress_level: float =  round(0.0000, 4)
        self._max_stress: float = 100.0

    def _register_events(self):
        self._event_notifier.register(ComponentEvent.DORMANCY_UPDATED, self._handle_dormancy_update)
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)
        self._event_notifier.register(ComponentEvent.EXTREME_WEATHER, self._handle_extreme_weather)

    def _handle_dormancy_update(self, *args, **kwargs):
        dormant: bool = kwargs.get("dormant", False)
        if self._is_dormant != dormant:
            self._is_dormant = dormant

    def _handle_stress_update(self, *args, **kwargs):
        new_stress: float = kwargs.get("stress_level", 0.0)
        self._stress_level = new_stress

    def _handle_extreme_weather(self, *args, **kwargs):
        temperature = kwargs.get("temperature", 0.0)
        self._logger.error(f"EXTREME WEATHER HANDLING FROM A COMPONENT: {temperature}")

        if temperature <= ClimateThresholds.Temperature.EXTREME_COLD:
            stress_change = ClimateThresholds.StressChange.EXTREME_COLD / 5.0
            self.modify_stress(stress_change, StressReason.TEMPERATURE_EXTREME)

        elif temperature <= ClimateThresholds.Temperature.COLD:
            stress_change = ClimateThresholds.StressChange.COLD / 5.0
            self.modify_stress(stress_change, StressReason.TEMPERATURE_EXTREME)

        elif ClimateThresholds.Temperature.OPTIMAL_LOW <= temperature <= ClimateThresholds.Temperature.OPTIMAL_HIGH:
            stress_change = ClimateThresholds.StressChange.OPTIMAL / 5.0
            self.modify_stress(stress_change, StressReason.TEMPERATURE_OPTIMAL)

        elif temperature >= ClimateThresholds.Temperature.EXTREME_HOT:
            stress_change = ClimateThresholds.StressChange.EXTREME_HOT / 5.0
            self.modify_stress(stress_change, StressReason.TEMPERATURE_EXTREME)

        elif temperature >= ClimateThresholds.Temperature.HOT:
            stress_change = ClimateThresholds.StressChange.HOT / 5.0
            self.modify_stress(stress_change, StressReason.TEMPERATURE_EXTREME)

    def request_dormancy(self, reason: DormancyReason, active: bool) -> None:
        if active:
            self._dormancy_reasons.add(reason)
        else:
            if not self._dormancy_reasons:
                return
            self._dormancy_reasons.discard(reason)

        self._event_notifier.notify(ComponentEvent.DORMANCY_REASONS_CHANGED,
                                    component=self.__class__,
                                    reasons=self._dormancy_reasons)

    def modify_stress(self, delta: float, reason: StressReason):
        self._logger.error(f"TRYING TO MODIFY STRESS LEVEL {delta} - Reason: {reason}")
        old_stress = self._stress_level
        self._stress_level = round(max(0.0, min(self._stress_level + delta, self._max_stress)), 4)

        self._logger.error(f"STRESS IN SSTRESS {delta}")
        if old_stress != self._stress_level:

            self._event_notifier.notify(
                ComponentEvent.STRESS_UPDATED,
                stress_level=self._stress_level,
                stress_delta=self._stress_level - old_stress,
                normalized_stress=self._stress_level / self._max_stress,
                reason=reason
            )

            # if self._stress_level > self._max_stress * 0.9 and not self._is_dormant:
            #     self.request_dormancy(DormancyReason.ENVIRONMENTAL_STRESS, True)
            # elif self._stress_level < self._max_stress * 0.7  and DormancyReason.ENVIRONMENTAL_STRESS in self._dormancy_reasons:
            #     self.request_dormancy(DormancyReason.ENVIRONMENTAL_STRESS, False)
