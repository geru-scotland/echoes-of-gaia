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
Abstract base class for entity attribute management handlers.

Provides common event notification infrastructure and logging setup;
defines abstract event registration interface for handlers.
Supports event notifier integration and handler lifecycle management.
"""

from abc import ABC, abstractmethod

from biome.systems.events.event_notifier import EventNotifier
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class AttributeHandler(ABC):

    def __init__(self, event_notifier: EventNotifier):
        self._event_notifier = event_notifier
        self._logger = LoggerManager.get_logger(Loggers.BIOME)

    @abstractmethod
    def register_events(self):
        pass

    def disable(self):
        self._event_notifier = None