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

"""
Base classes for component system architecture.

Defines Component and EntityComponent abstract base classes;
handles event notification, dormancy management and host references.
Provides foundation for modular entity behavior implementation - supports
lifecycle management and state tracking across component hierarchy.
"""

from abc import abstractmethod
from logging import Logger
from typing import Any, Dict, Set

from simpy import Environment as simpyEnv

from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason
from shared.enums.strings import Loggers
from shared.events.handler import EventHandler
from utils.loggers import LoggerManager


class Component(EventHandler):
    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier):
        self._event_notifier: EventNotifier = event_notifier
        super().__init__()
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._type: ComponentType = type
        self._env: simpyEnv = env
        self._host_alive: bool = True

    def disable_notifier(self):
        self._host_alive = False
        self._event_notifier = None

    @abstractmethod
    def _register_events(self):
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def type(self):
        return self._type


class EntityComponent(Component):
    def __init__(self, env: simpyEnv, type: ComponentType, event_notifier: EventNotifier, lifespan: float = 10.0):

        super().__init__(env, type, event_notifier)

        self._host = None
        self._dormancy_reasons: Set[DormancyReason] = set()
        self._is_dormant: bool = False
        self._lifespan: float = lifespan

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
            if not self._dormancy_reasons:
                return
            self._dormancy_reasons.discard(reason)

        self._event_notifier.notify(ComponentEvent.DORMANCY_REASONS_CHANGED,
                                    component=self.__class__,
                                    reasons=self._dormancy_reasons)

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_host(self, ref):
        self._host = ref
