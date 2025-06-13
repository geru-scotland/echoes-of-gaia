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

"""
Coordinates InfluxDB operations and event handling lifecycle.

Integrates InfluxDB client with event handlers for complete
telemetry pipeline management; provides unified interface
for database initialization and cleanup operations.
"""

from typing import Dict, Any

from simulation.core.systems.telemetry.event_handler import InfluxEventHandler
from simulation.core.systems.telemetry.influxdb import InfluxDB


class InfluxDBManager:
    def __init__(self, config: Dict[str, Any]):
        self._influxdb: InfluxDB = InfluxDB(config)
        self._event_handler: InfluxEventHandler = InfluxEventHandler(self._influxdb)

    def start(self) -> None:
        self._influxdb.listen()

    def close(self):
        self._influxdb.close()