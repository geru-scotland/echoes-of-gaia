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

from shared.strings import Loggers
from utils.loggers import LoggerManager


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
