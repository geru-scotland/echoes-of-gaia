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
from typing import List, Dict, Any

from pandas import DataFrame
from simpy import Environment as simpyEnv

from biome.systems.climate.system import ClimateSystem
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.history.climate_history import ClimateHistoryService
from shared.enums.events import BiomeEvent
from shared.enums.strings import Loggers
from shared.timers import Timers
from utils.loggers import LoggerManager


class ClimateDataManager:
    def __init__(self, env: simpyEnv, climate_service: ClimateSystem):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._climate_service = climate_service
        self._collector = ClimateDataCollector(climate_service)
        self._climate_history = ClimateHistoryService()
        self._env: simpyEnv = env
        self._evolution_cycle: int = 0

    def record_daily_data(self) -> None:
        self._collect_daily()

    def set_evolution_cycle(self, evolution_cycle: int):
        self._evolution_cycle = evolution_cycle

    def _collect_daily(self):
        climate_data: Dict[str, Any] = self._collector.collect_data()
        self._climate_history.add_daily_data(climate_data, self._evolution_cycle, self._env.now)

    def get_data(self, evolution_cycle: int) -> DataFrame:
        return self._climate_history.get_data_by_evolution_cycle(evolution_cycle)

    def get_climate_history_service(self):
        return self._climate_history

    def get_current_month_averages(self) -> Dict[str, float]:
        return self._climate_history.get_current_month_averages()

    def get_climate_system(self) -> ClimateSystem:
        return self._climate_service
