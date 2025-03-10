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
from shared.enums.reasons import DormancyReason
from shared.enums.strings import Loggers
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

    def _register_events(self):
        self._event_notifier.register(ComponentEvent.DORMANCY_UPDATED, self._handle_dormancy_update)

    def _handle_dormancy_update(self, *args, **kwargs):
        dormant: bool = kwargs.get("dormant", False)
        if self._is_dormant != dormant:
            self._is_dormant = dormant

    def request_dormancy(self, reason: DormancyReason, active: bool) -> None:
        if active:
            self._dormancy_reasons.add(reason)
        else:
            self._dormancy_reasons.discard(reason)

        self._event_notifier.notify(ComponentEvent.DORMANCY_REASONS_CHANGED,
                                    component=self.__class__,
                                    reasons=self._dormancy_reasons)