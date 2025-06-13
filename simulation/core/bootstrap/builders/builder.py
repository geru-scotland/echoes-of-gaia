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
Abstract base classes for bootstrap builder pattern implementation.

Defines builder interface for context construction and configurator
strategy pattern for flexible component setup; enables extensible
bootstrap architecture with consistent initialization patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from config.settings import Settings
from simulation.core.bootstrap.context.context import Context


class ConfiguratorStrategy(ABC):
    @abstractmethod
    def configure(self, settings: Settings, **kwargs: Any) -> None:
        raise NotImplementedError


class Builder(ABC):
    def __init__(self):
        self._context: Optional[Context] = None

    def _initialise(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    @property
    def context(self) -> Context:
        return self._context


