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
import itertools
import sys
import time
import traceback
from logging import Logger
from typing import Optional, cast, Tuple, Dict, Any

import simpy

from biome.api.biome_api import BiomeAPI
from biome.biome import Biome
from biome.services.biome_control_service import BiomeControlService
from biome.systems.managers.biome_data_manager import BiomeDataManager
from config.settings import Settings
from shared.enums.enums import SimulationMode
from shared.enums.events import SimulationEvent
from shared.timers import Timers
from shared.enums.strings import Strings, Loggers
from simulation.core.bootstrap.bootstrap import Bootstrap
from simulation.core.bootstrap.context.context import Context
from simulation.core.bootstrap.context.context_data import BiomeContextData, SimulationContextData
from simulation.core.experiment_path_manager import ExperimentPathManager
from simulation.core.systems.events.event_bus import SimulationEventBus
from simulation.core.systems.telemetry.datapoint import Datapoint
from simulation.core.systems.time.time import SimulationTime
from utils.loggers import LoggerManager
from utils.middleware import log_execution_time


class SimulationEngine:
    @log_execution_time(context="Biome loading")
    def __init__(self, settings: Settings, mode: SimulationMode):
        self._env: simpy.Environment = simpy.Environment()
        self._id_generator = itertools.count(0)
        try:
            self._simulation_mode: SimulationMode = mode
            biome_context, simulation_context = self._boot_and_get_contexts(settings)
            self._context = simulation_context

            self._eras = self._context.config.get("eras", {}).get("amount", 0)
            self._events_per_era = self._context.config.get("eras", {}).get("events-per-era", 0)
            self._datapoints: bool = self._context.config.get("datapoints", False)
            self._snapshots: bool = self._context.config.get("data", {}).get("snapshots", {}).get("enabled", False)
            self._dataset_generation: bool = self._context.config.get("data", {}).get("dataset", {}).get("generate",
                                                                                                         False)

            data_storage: Dict[str, Any] = self._context.config.get("data", {}).get("storage", {})
            trackers: Dict[str, Any] = data_storage.get("trackers", {})

            sim_paths: Dict[str, Any] = {
                "base": data_storage.get("base", ""),
                "simulations": data_storage.get("simulations", ""),
                "evolution": trackers.get("evolution", ""),
                "genetic_crossover": trackers.get("genetic_crossover", ""),
                "trends": trackers.get("trends", ""),
            }

            options: Dict[str, bool] = {
                "smart-population": self._context.config.get("population-control", {}).get("smart", False),
                "smart_plot": self._context.config.get("data", {}).get("visualization", {}).get("smart-trends", False),
                "evolution_tracking": self._context.config.get("data", {}).get("visualization", {}).get("evolution",
                                                                                                        False),
                "crossover_tracking": self._context.config.get("data", {}).get("visualization", {}).get("crossover",
                                                                                                        False),
                "remove_dead_entities": self._context.config.get("cleanup", {}).get("remove_dead_entities", False)
            }

            if any([
                bool(options["evolution_tracking"]),
                bool(options["crossover_tracking"]),
                bool(options["smart_plot"])
            ]):
                experiment_path_manager: ExperimentPathManager = ExperimentPathManager.get_instance()
                experiment_path_manager.initialize(sim_paths)

            self._logger: Logger = LoggerManager.get_logger(Loggers.SIMULATION)
            self._biome_api = BiomeAPI(biome_context, self._env, self._simulation_mode, options)
            BiomeControlService.get_instance().initialize(self._biome_api)

            if self._datapoints:
                self._context.influxdb.start()

            if self._snapshots:
                self._data_manager = BiomeDataManager(
                    env=self._env,
                    config=self._context.config,
                    dataset_generation=self._dataset_generation
                )

                self._data_manager.configure(self._biome_api.get_biome())

            self._time: SimulationTime = SimulationTime(self._events_per_era)

            SimulationEventBus.trigger("biome_loaded", biome_context.tile_map)
        except Exception as e:
            self._logger = LoggerManager.get_logger(Loggers.BOOTSTRAP)
            self._logger.exception(f"[Simulation Engine] There was an error bootstraping: {e}")
            tb = traceback.format_exc()
            self._logger.error("Error: %s", tb)

    def _boot_and_get_contexts(self, settings: Settings) -> Tuple[BiomeContextData, SimulationContextData]:
        bootstrap: Bootstrap = Bootstrap(settings)
        context: Optional[Context] = bootstrap.get_context()
        biome_context = cast(BiomeContextData, context.get(Strings.BIOME_CONTEXT))
        simulation_context = cast(SimulationContextData, context.get(Strings.SIMULATION_CONTEXT))
        return biome_context, simulation_context

    def _montly_update(self, timer: int):
        yield self._env.timeout(timer)
        while True:
            self._logger.debug("Monthly state log.")

            if self._datapoints:
                simulated_timestamp = int(time.time() * 1000)
                datapoint_id = next(self._id_generator)
                biome_datapoint: Optional[Datapoint] = self._data_manager.collect_data_for_telemetry(
                    datapoint_id, simulated_timestamp
                )
                if biome_datapoint:
                    SimulationEventBus.trigger("on_biome_data_collected", biome_datapoint)

            self._time.log_time(self._env.now)
            yield self._env.timeout(timer)

    @log_execution_time(context="Simulation executed in")
    def run(self):
        self._logger.info("Running simulation...")
        self._env.process(self._montly_update(Timers.Calendar.MONTH))
        self._time.log_time(self._env.now)
        self._env.run(until=self._eras * self._events_per_era)

        SimulationEventBus.trigger(SimulationEvent.SIMULATION_FINISHED)

        if self._datapoints:
            self._context.influxdb.close()

        if self._snapshots and self._data_manager:
            self._data_manager.shutdown()

        self._time.log_time(self._env.now)

    def shutdown_training(self):
        self._logger.info("Shutting down training")
        SimulationEventBus.trigger(SimulationEvent.SIMULATION_FINISHED)

        if self._datapoints:
            self._context.influxdb.close()

        if self._snapshots and self._data_manager:
            self._data_manager.shutdown()

        self._time.log_time(self._env.now)

    def step(self, time_delta: int = 1) -> int:
        current_time = self._env.now

        self._env.run(until=current_time + time_delta)
        self._time.log_time(self._env.now)

        return self._env.now

    def get_biome(self) -> Biome:
        return self._biome_api.get_biome()
