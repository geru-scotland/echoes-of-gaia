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
import math
from logging import Logger
from abc import abstractmethod, ABC
from typing import Optional, Set, Dict, Any

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
        self._host_alive: bool = True

    def disable_notifier(self):
        self._host_alive = False
        self._event_notifier = None

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
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
        self._host = None

    @abstractmethod
    def _register_events(self):
       raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_host(self, ref):
        self._host = ref

    def _update(self, delay: Optional[int] = None):
        pass

class FloraComponent(EntityComponent):

    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier, lifespan: float):
        super().__init__(env, type, event_notifier)
        self._event_notifier: EventNotifier = event_notifier
        self._dormancy_reasons: Set[DormancyReason] = set()
        self._is_dormant: bool = False
        self._stress_level: float =  round(0.0000, 4)
        self._max_stress: float = 100.0
        self._lifespan: float = lifespan

    def _register_events(self):
        self._event_notifier.register(ComponentEvent.DORMANCY_UPDATED, self._handle_dormancy_update)
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)

    def _handle_dormancy_update(self, *args, **kwargs):
        dormant: bool = kwargs.get("dormant", False)
        if self._is_dormant != dormant:
            self._is_dormant = dormant

    def _handle_stress_update(self, *args, **kwargs):
        new_stress: float = kwargs.get("stress_level", 0.0)
        self._stress_level = new_stress

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
        self._logger.debug(f"[{id(self._host)}|{str(self._host._descriptor.species).upper()} {self.__class__} STRESS DEBUG] Attempting to modify stress level.")
        self._logger.debug(f"[STRESS DEBUG] - Delta: {delta}")
        self._logger.debug(f"[STRESS DEBUG] - Reason: {reason}")
        self._logger.debug(f"[STRESS DEBUG] - Previous Stress Level: {self._stress_level}")
        self._logger.debug(f"[STRESS DEBUG] - Max Stress Allowed: {self._max_stress}")

        lifespan_factor = 1.0 / math.sqrt(max(1.0, self._lifespan))

        old_stress = self._stress_level
        new_stress = max(0.0, min(self._stress_level + delta * lifespan_factor, self._max_stress))
        rounded_stress = round(new_stress, 4)

        self._logger.debug(f"[STRESS DEBUG] - New Calculated Stress (Before Rounding): {new_stress}")
        self._logger.debug(f"[STRESS DEBUG] - New Stress Level (Rounded): {rounded_stress}")

        self._stress_level = rounded_stress
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

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError


class EnergyBasedFloraComponent(FloraComponent):
    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier, lifespan: float,
                 max_energy_reserves: float = 100.0):
        super().__init__(env, type, event_notifier, lifespan)

        self._energy_reserves: float = round(max_energy_reserves, 2)
        self._max_energy_reserves: float = round(max_energy_reserves, 2)

    def _register_events(self):
        super()._register_events()
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)

    def _handle_energy_update(self, *args, **kwargs):
        new_energy: float = kwargs.get("energy_reserves", 0.0)
        self._energy_reserves = new_energy

    def modify_energy(self, energy_delta: float) -> None:
        old_energy: float = self._energy_reserves

        self._energy_reserves = max(0.0, min(self._energy_reserves + energy_delta, self._max_energy_reserves))

        if old_energy != self._energy_reserves:
            self._event_notifier.notify(ComponentEvent.ENERGY_UPDATED,
                                        energy_reserves=self._energy_reserves)
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
       raise  NotImplementedError