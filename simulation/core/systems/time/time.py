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
from dataclasses import dataclass
from typing import Dict

from shared.strings import Loggers
from utils.loggers import LoggerManager


@dataclass
class SimulationTimeInfo:
    raw_ticks: int
    month: int
    year: int

    @classmethod
    def from_ticks(cls, ticks: int) -> 'SimulationTimeInfo':
        """
        Convierte ticks en una estructura de tiempo de simulación.
        Los meses empiezan en 1, los años en 0.
        """
        ticks_per_month = 60
        months_per_year = 12

        total_months = ticks // ticks_per_month
        years = (total_months - 1) // months_per_year
        months = ((total_months - 1) % months_per_year) + 1


        return cls(
            raw_ticks=ticks,
            month=months,
            year=years,
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "raw_ticks": self.raw_ticks,
            "month": self.month,
            "year": self.year,
            "total_months": (self.year * 12) + self.month
        }

class SimulationTime:
    def __init__(self, events_per_era: int = 30000):
        self._logger: logging.Logger = LoggerManager.get_logger(Loggers.SIMULATION)
        self.events_per_era = events_per_era

    def get_current_era(self, event_count: int) -> int:
        return event_count // self.events_per_era

    def get_event_number_in_era(self, event_count: int) -> int:
        return event_count % self.events_per_era

    def log_time(self, event_count: int):
        era = self.get_current_era(event_count)
        event_number = self.get_event_number_in_era(event_count)
        self._logger.info(f"Simulated events: {event_count}, Era={era}, Event in Era={event_number}")
