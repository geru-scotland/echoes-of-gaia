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
from simulation.core.systems.metrics.event_handler import InfluxEventHandler
from simulation.core.systems.metrics.influxdb import InfluxDB


class InfluxDBManager:
    def __init__(self):
        self._influxdb: InfluxDB = InfluxDB()
        self._event_handler: InfluxEventHandler = InfluxEventHandler(self._influxdb)

    def start(self) -> None:
        self._influxdb.listen()