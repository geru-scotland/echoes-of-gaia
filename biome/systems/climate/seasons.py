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
from typing import List, Callable

from shared.enums import Season
from shared.strings import Loggers
from utils.loggers import LoggerManager


class SeasonSystem:
    def __init__(self, initial_season: Season):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._day: int = 0
        self._current_season: Season = initial_season
        self._next_season_day: int = 0
        self._seasons: List[Season] = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
        self._next_season_day: int = 90

    def _advance_season(self) -> None:
        self._logger.debug(f"Advancing season, day: {self._day}")
        try:
            current_season_idx: int = self._seasons.index(self._current_season)
            self._current_season = self._seasons[(current_season_idx + 1) % len(self._seasons)]
            self._logger.info(f"Current season: {self._current_season}")
            # Por ahora, simplemente incremento en 90 por cada season, ya pondré duración
            # determinada por season, no hace falta acotar en el año al ser ticks
            self._next_season_day += 90
        except Exception as e:
            self._logger.exception(f"There was an error advancing the season: {e}")

    def update(self, handle_new_season: Callable, days: int = 1) -> None:
        self._day += days
        if self._day == self._next_season_day:
            self._advance_season()
            handle_new_season(self._current_season)

    def get_current_season(self) -> Season:
        return self._current_season