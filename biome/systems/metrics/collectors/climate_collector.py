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
import time
from logging import Logger
from typing import Dict, Any

from biome.systems.climate.system import ClimateSystem
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class ClimateDataCollector:
    def __init__(self, climate_service: ClimateSystem):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._climate_service: ClimateSystem = climate_service

    def collect_data(self) -> Dict[str, Any]:
        self._logger.debug("Recolectando datos climáticos actuales")

        climate_state = self._climate_service.get_state()
        if not climate_state:
            return self._get_empty_climate_data()

        climate_data = {
            "temperature": climate_state.temperature,
            "humidity": climate_state.humidity,
            "precipitation": climate_state.precipitation,
            "atm_pressure": climate_state.atm_pressure,
            "current_season": str(self._climate_service.get_current_season()),
            "timestamp": int(time.time())
        }

        return climate_data

    def _get_empty_climate_data(self) -> Dict[str, Any]:
        return {
            "temperature": 0.0,
            "humidity": 0.0,
            "precipitation": 0.0,
            "atm_pressure": 1013.0,
            "current_season": "unknown",
            "timestamp": int(time.time())
        }

    # metricas. Pongo aquí, para que luego el sistema de snapshots pueda utilizar también
    # así, luego tomar datos y mostrar en viewer, algo como: last evolution cycle (y mostrar
    # estadísticas medias del clima
    # recolectar para no solo average, si no también para min, y max temperatura, humedad, prec...
    # guardar historial y sacar most common weather event etc, para utilizar en la evolución.