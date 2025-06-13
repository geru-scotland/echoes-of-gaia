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
Abstract state handler base class for state management operations.

Provides state dictionary management with abstract state computation;
includes JSON serialization support for state data export.
Defines standard interface for state computation and persistence.
"""

import json
from abc import ABC, abstractmethod


class StateHandler(ABC):
    def __init__(self):
        self._state = {}

    @abstractmethod
    def compute_state(self):
        raise NotImplementedError

    def dump_state(self) -> str:
        return json.dumps(self.compute_state(), indent=2)
