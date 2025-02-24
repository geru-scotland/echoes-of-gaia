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
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class EntityState:
    values: Dict[str, Any] = field(default_factory=dict)

    def update(self, key: str, value: Any):
        self.values[key] = value

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def dump(self) -> Dict[str, Any]:
        return self.values.copy()