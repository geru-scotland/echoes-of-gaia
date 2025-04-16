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
from abc import ABC, abstractmethod
from typing import Optional, Dict

from shared.types import PredictionResult, SymbolicResult, IntegratedResult


class IntegrationStrategy(ABC):
    @abstractmethod
    def integrate(self,
                  neural_result: PredictionResult,
                  symbolic_result: SymbolicResult,
                  confidence_weights: Optional[Dict[str, float]] = None) -> IntegratedResult:
        raise NotImplementedError
