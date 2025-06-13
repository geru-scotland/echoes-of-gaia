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
Manages context data storage with type-safe operations.

Provides centralized container for ContextData objects;
enforces type validation and key-based retrieval patterns.
Enables organized access to simulation and biome contexts.
"""

from logging import Logger
from typing import Any, Dict, Optional

from simulation.core.bootstrap.context.context_data import ContextData


class Context:
    def __init__(self, logger: Logger) -> None:
        self._logger: Logger = logger
        self._logger.info("[Context] Creating context.")
        self.data: Optional[Dict[str, ContextData]] = {}

    def get(self, key: str, default: Any = None) -> ContextData:
        return self.data.get(key, default)

    def set(self, key: str, value: ContextData) -> None:
        if not isinstance(value, ContextData):
            raise TypeError(f"Expected ContextData, got {type(value).__name__}")
        self.data[key] = value

    def __repr__(self) -> str:
        return f"Context({self.data})"