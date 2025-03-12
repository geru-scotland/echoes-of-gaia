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
from logging import Logger
from typing import List, Dict, Any, Optional

from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class ClimateHistoryService:
    def __init__(self, max_history_size: int = 100):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._history: List[Dict[str, Any]] = []
        self._max_history_size: int = max_history_size

    def add_climate_data(self, climate_data: Dict[str, Any]) -> None:
        self._history.append(climate_data)

        if len(self._history) > self._max_history_size:
            self._history.pop(0)

    def get_history(self, num_entries: Optional[int] = None) -> List[Dict[str, Any]]:
        if num_entries is None:
            return self._history.copy()
        return self._history[-min(num_entries, len(self._history)):]

    def get_average_over_period(self, period: int) -> Dict[str, float]:
        recent_data: List[Dict[str, Any]] = self.get_history(period)
        if not recent_data:
            return {}

        totals: Dict[str, float] = {key: 0.0 for key in recent_data[0].keys()
                  if isinstance(recent_data[0][key], (int, float))}

        for entry in recent_data:
            for key in totals:
                if key in entry and isinstance(entry[key], (int, float)):
                    totals[key] += entry[key]

        averages = {key: value / len(recent_data) for key, value in totals.items()}
        return averages

    def clear_history(self) -> None:
        self._history.clear()