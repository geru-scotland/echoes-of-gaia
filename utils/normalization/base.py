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
from typing import TypeVar, Generic, Tuple

T = TypeVar('T')
V = TypeVar('V')


class Normalizer(Generic[T, V], ABC):
    @abstractmethod
    def normalize(self, value: T) -> V:
        pass

    @abstractmethod
    def denormalize(self, normalized_value: V) -> T:
        pass

    @abstractmethod
    def get_range(self) -> Tuple[T, T]:
        pass