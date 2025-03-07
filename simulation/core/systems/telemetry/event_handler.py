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
from simulation.core.systems.events.event_bus import GlobalEventBus
from simulation.core.systems.events.handler import EventHandler
from simulation.core.systems.telemetry.datapoint import Datapoint
from simulation.core.systems.telemetry.influxdb import InfluxDB


class InfluxEventHandler(EventHandler):
   def __init__(self, influxdb: InfluxDB):
       self._influxdb: InfluxDB = influxdb
       super().__init__()

   def _register_events(self):
       GlobalEventBus.register("on_biome_data_collected", self._influxdb.write_data_point)

   def on_data_collected(self, datapoint: Datapoint):
       self._influxdb.write_data_point(datapoint)



