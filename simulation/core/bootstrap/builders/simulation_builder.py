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
from logging import Logger
from typing import Optional

from config.settings import Config, SimulationSettings
from shared.strings import Loggers
from simulation.core.bootstrap.context.context_data import SimulationContextData
from simulation.core.bootstrap.builders.builder import Builder
from simulation.core.systems.telemetry.manager import InfluxDBManager
from utils.loggers import LoggerManager


class SimulationBuilder(Builder):
    def __init__(self, settings: SimulationSettings):
        super().__init__()
        self._settings = settings
        self._logger: Logger = LoggerManager.get_logger(Loggers.SIMULATION)
        self._logger.info("[Simulation Builder] Initialising SimulationBuilder...")
        self._context_data: Optional[SimulationContextData] = None
        self._initialise()

    def _initialise(self):
        pass

    def build(self) -> None:
        self._logger.info("[Simulation Builder] Simulation builder...")
        try:
            config: Config = self._settings.config.get("simulation")
            influxdb_mgr: InfluxDBManager = InfluxDBManager(self._settings.influxdb_config)
            self._context = SimulationContextData(config=config, logger_name=Loggers.SIMULATION,
                                                  influxdb=influxdb_mgr)
        except Exception as e:
            self._logger.exception(f"There was a problem building the context from the Simulation: {e}")