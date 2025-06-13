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
from typing import Dict, Tuple, Union

from .base import Normalizer
from .ranges import CLIMATE_RANGES

Number = Union[int, float]


class MinMaxNormalizer(Normalizer[Number, Number]):
    def __init__(self, min_value: Number, max_value: Number):
        self._min = min_value
        self._max = max_value

        if self._min == self._max:
            self._range = 1.0
        else:
            self._range = self._max - self._min

    def normalize(self, value: Number) -> Number:
        if self._range == 0:
            return 0.5

        clamped = max(self._min, min(self._max, value))
        return (clamped - self._min) / self._range

    def denormalize(self, normalized_value: Number) -> Number:
        clamped = max(0.0, min(1.0, normalized_value))
        return self._min + (clamped * self._range)

    def get_range(self) -> Tuple[Number, Number]:
        return self._min, self._max


class DomainNormalizer:
    def __init__(self, domain_ranges: Dict[str, Tuple[Number, Number]]):
        self._normalizers: Dict[str, MinMaxNormalizer] = {}

        for var_name, (min_val, max_val) in domain_ranges.items():
            self._normalizers[var_name] = MinMaxNormalizer(min_val, max_val)

    def normalize(self, variable: str, value: Number) -> Number:
        if variable not in self._normalizers:
            raise ValueError(f"Variable '{variable}' not defined")

        return self._normalizers[variable].normalize(value)

    def denormalize(self, variable: str, normalized_value: Number) -> Number:
        if variable not in self._normalizers:
            raise ValueError(f"Variable '{variable}' not defined")

        return self._normalizers[variable].denormalize(normalized_value)

    def get_range(self, variable: str) -> Tuple[Number, Number]:
        if variable not in self._normalizers:
            raise ValueError(f"Variable '{variable}' not defined")

        return self._normalizers[variable].get_range()

    def get_variables(self) -> list:
        return list(self._normalizers.keys())

climate_normalizer = DomainNormalizer(CLIMATE_RANGES)