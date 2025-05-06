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
import logging
import time
from dataclasses import dataclass
from typing import Dict

from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


@dataclass
class SimulationTimeInfo:
    raw_ticks: int
    month: int
    year: int

    @classmethod
    def from_ticks(cls, ticks: int):
        from shared.timers import Timers

        month = (ticks // Timers.Calendar.MONTH) % 12
        year = ticks // (12 * Timers.Calendar.MONTH)

        if month == 0:
            month = 12
            if year > 0:
                year -= 1

        year = max(0, year)

        if ticks < Timers.Calendar.MONTH:
            month = 1

        return cls(
            raw_ticks=ticks,
            month=month,
            year=year
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_ticks": self.raw_ticks,
            "month": self.month,
            "year": self.year,
        }


class SimulationTime:
    def __init__(self, events_per_era: int = 30000):
        self._logger: logging.Logger = LoggerManager.get_logger(Loggers.SIMULATION)
        self.events_per_era = events_per_era
        self._start_time = time.time()

    def get_current_era(self, event_count: int) -> int:
        return event_count // self.events_per_era

    def get_event_number_in_era(self, event_count: int) -> int:
        return event_count % self.events_per_era

    def log_time(self, event_count: int):
        era = self.get_current_era(event_count)
        event_number = self.get_event_number_in_era(event_count)
        elapsed_seconds = time.time() - self._start_time
        # TODO: Si training mode, log debug si no, info
        self._logger.info(
            f"Simulated events: {event_count}, Era: {era}, Event in Era: {event_number}"
            f" in {elapsed_seconds:.2f} seconds")
