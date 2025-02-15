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
import time
from logging import Logger
from typing import Optional, cast, Tuple

import simpy

from biome.api.biome_api import BiomeAPI
from config.settings import Settings
from shared.enums import Timers
from shared.strings import Strings
from simulation.core.bootstrap.bootstrap import Bootstrap
from simulation.core.bootstrap.context.context import Context
from simulation.core.bootstrap.context.context_data import BiomeContextData, SimulationContextData
from simulation.core.systems.events.dispatcher import EventDispatcher
from simulation.core.systems.time.time import SimulationTime
from utils.loggers import setup_logger


class SimulationEngine:
    def __init__(self, settings: Settings):
        self._env: simpy.Environment = simpy.Environment()
        # TODO: El logger tiene que ser cargado por el builder y el contexto
        try:
            biome_context, simulation_context = self._boot_and_get_contexts(settings)
            self._biome_api = BiomeAPI(biome_context, self._env)
            self._context = simulation_context
            self._logger: Logger = self._context.logger
            self._eras = self._context.config.get("eras", {}).get("amount", 0)
            self._events_per_era = self._context.config.get("eras", {}).get("events-per-era", 0)
            self._time: SimulationTime = SimulationTime(self._events_per_era)
            EventDispatcher.trigger("biome_loaded", biome_context.map)
        except Exception as e:
            self._logger = setup_logger("bootstrap", "bootstrap.log")
            self._logger.exception(f"[Simulation Engine] There was an error bootstraping: {e}")

    def _boot_and_get_contexts(self, settings: Settings) -> Tuple[BiomeContextData, SimulationContextData]:
        bootstrap: Bootstrap = Bootstrap(settings)
        context: Optional[Context] = bootstrap.get_context()
        biome_context = cast(BiomeContextData, context.get(Strings.BIOME_CONTEXT))
        simulation_context = cast(SimulationContextData, context.get(Strings.SIMULATION_CONTEXT))
        return biome_context, simulation_context

    def _montly_update(self, timer: int):
        """
         TODO: DECORATOR para este tipo de updataes.
         TODO 2: Hacer SNAPSHOT AQUI, CADA X TIEMPO.
         Con biome api, hacer que haga un compute state o snapshot.
        """
        yield self._env.timeout(timer)
        while True:
            self._logger.info("[SIMULATION] Monthly update.")
            self._time.log_time(self._env.now)
            yield self._env.timeout(timer)

    def run(self):
        self._logger.info("Running simulation...")
        self._env.process(self._montly_update(Timers.Simulation.MONTH))
        self._time.log_time(self._env.now)
        self._env.run(until=self._eras * self._events_per_era)
        self._time.log_time(self._env.now)

