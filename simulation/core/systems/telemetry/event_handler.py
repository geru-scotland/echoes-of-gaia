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
Bridges event system with InfluxDB telemetry storage.

Registers event handlers for biome data collection events;
routes telemetry datapoints to InfluxDB writer automatically.
Connects simulation events to persistent data storage layer.
"""

from shared.events.handler import EventHandler
from simulation.core.systems.events.event_bus import SimulationEventBus
from simulation.core.systems.telemetry.datapoint import Datapoint
from simulation.core.systems.telemetry.influxdb import InfluxDB


class InfluxEventHandler(EventHandler):
   def __init__(self, influxdb: InfluxDB):
       self._influxdb: InfluxDB = influxdb
       super().__init__()

   def _register_events(self):
       SimulationEventBus.register("on_biome_data_collected", self._influxdb.write_data_point)

   def on_data_collected(self, datapoint: Datapoint):
       self._influxdb.write_data_point(datapoint)



