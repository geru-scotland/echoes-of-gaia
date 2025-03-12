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

from simpy import Environment as simpyEnv

from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.history.climate_history import ClimateHistoryService
from shared.enums.events import BiomeEvent
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class ClimateDataManager:
    def __init__(self, climate_service, collection_interval: int = 60):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._climate_service = climate_service
        self._collector = ClimateDataCollector(climate_service)
        self._history_service = ClimateHistoryService()
        self._collection_interval = collection_interval
        self._env = None

    def start(self, env: simpyEnv) -> None:
        self._env = env
        self._env.process(self._collect_periodically())

    def _collect_periodically(self):
        while True:
            climate_data = self._collector.collect_data()
            self._history_service.add_climate_data(climate_data)
            self._logger.debug(f"Datos climáticos recolectados: {climate_data}")

            recent_history = self.get_recent_climate_history(5)
            climate_averages = self.get_climate_averages(10)
            latest_climate_data = self.get_latest_climate_data()

            self._logger.info(
                f"[Periodic Check] Nuevo dato climático recolectado: {climate_data}\n"
                f"  - Últimos 5 registros: {recent_history}\n"
                f"  - Promedios climáticos (últimos 10 períodos): {climate_averages}\n"
                f"  - Último dato climático registrado: {latest_climate_data}"
            )
            BiomeEventBus.trigger(BiomeEvent.CLIMATE_DATA_COLLECTED, climate_data)

            yield self._env.timeout(self._collection_interval)

    def get_recent_climate_history(self, num_entries: int) -> List[Dict[str, Any]]:
        return self._history_service.get_history(num_entries)

    def get_climate_averages(self, period: int) -> Dict[str, float]:
        return self._history_service.get_average_over_period(period)

    def get_latest_climate_data(self) -> Dict[str, Any]:
        history = self._history_service.get_history(1)
        if history:
            return history[0]
        return self._collector.collect_data()

    def shutdown(self) -> None:
        self._history_service.clear_history()