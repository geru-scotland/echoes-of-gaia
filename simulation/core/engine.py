import time
from logging import Logger
from typing import Optional, cast, Tuple

import simpy

from biome.api.biome_api import BiomeAPI
from config.settings import Settings
from shared.enums import Strings
from simulation.core.bootstrap.bootstrap import Bootstrap
from simulation.core.bootstrap.context.context import Context
from simulation.core.bootstrap.context.context_data import BiomeContextData, SimulationContextData
from simulation.core.systems.events.dispatcher import EventDispatcher
from utils.loggers import setup_logger


class SimulationEngine:
    def __init__(self, settings: Settings):
        self._env: simpy.Environment = simpy.Environment()
        # TODO: El logger tiene que ser cargado por el builder y el contexto
        try:
            biome_context, simulation_context = self._boot_and_get_contexts(settings)
            self._biome_api = BiomeAPI(biome_context)
            self._context = simulation_context
            self._logger: Logger = self._context.logger
            EventDispatcher.dispatch("biome_loaded", biome_context.map)
        except Exception as e:
            self._logger = setup_logger("bootstrap", "bootstrap.log")
            self._logger.exception(f"[Simulation Engine] There was an error bootstraping: {e}")

    def _boot_and_get_contexts(self, settings: Settings) -> Tuple[BiomeContextData, SimulationContextData]:
        bootstrap: Bootstrap = Bootstrap(settings)
        context: Optional[Context] = bootstrap.get_context()
        biome_context = cast(BiomeContextData, context.get(Strings.BIOME_CONTEXT))
        simulation_context = cast(SimulationContextData, context.get(Strings.SIMULATION_CONTEXT))
        return biome_context, simulation_context

    def run(self):
        self._logger.info("Running simulation...")
        print(self._context)
        self._env.process(self.step())

    def step(self):
        yield self._env.timeout(1)