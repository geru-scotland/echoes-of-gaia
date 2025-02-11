from typing import Dict, Any

from shared.enums import Strings
from config.settings import Settings
from simulation.core.bootstrap.builders.biome_builder import BiomeBuilder
from simulation.core.bootstrap.builders.simulation_builder import SimulationBuilder
from simulation.core.bootstrap.context.context import Context
from utils.exceptions import BootstrapError
from utils.loggers import setup_logger


class Bootstrap:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._logger = setup_logger("bootstrap", "bootstrap.log")
        self._context: Context = Context(self._logger)
        self._builders: Dict[str, Any] = {}
        self._build_context()

    def _setup_builders(self):
         self._builders[Strings.BIOME_BUILDER] = BiomeBuilder(self._settings.biome_settings, self._logger)
         self._builders[Strings.SIMULATION_BUILDER] = SimulationBuilder(self._settings.simulation_settings, self._logger)

    def _build(self):
         self._builders[Strings.BIOME_BUILDER].build()
         self._builders[Strings.SIMULATION_BUILDER].build()

    def _build_context(self):
        try:
            self._setup_builders()
            self._build()
            self._context.set(Strings.BIOME_CONTEXT, self._builders[Strings.BIOME_BUILDER].context)
            self._context.set(Strings.SIMULATION_CONTEXT, self._builders[Strings.SIMULATION_BUILDER].context)
            # TODO: Repetir para el sim context
        except Exception as e:
            raise BootstrapError(f"There was an error building the context: {e}")

    def get_context(self) -> Context:
        return self._context
