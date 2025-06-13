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

"""
Initializes simulation contexts through specialized builders.

Coordinates biome and simulation setup via dedicated builders;
loads bioma store data and constructs unified contexts.
Handles bootstrap failures with proper error recovery - ensures
reliable initialization before simulation execution begins.
"""

import traceback
from typing import Any, Dict

from config.settings import Settings
from exceptions.custom import BootstrapError, MapGenerationError
from shared.enums.strings import Strings
from shared.stores.biome_store import BiomeStore
from simulation.core.bootstrap.builders.biome_builder import BiomeBuilder
from simulation.core.bootstrap.builders.simulation_builder import SimulationBuilder
from simulation.core.bootstrap.context.context import Context
from utils.loggers import LoggerManager


class Bootstrap:
    def __init__(self, settings: Settings):
        self._settings: Settings = settings
        self._logger = LoggerManager.get_logger("bootstrap")
        self._context: Context = Context(self._logger)
        self._builders: Dict[str, Any] = {}
        BiomeStore.load_ecosystem_data()
        self._build_context()

    def _setup_builders(self):
        self._builders[Strings.BIOME_BUILDER] = BiomeBuilder(self._settings.biome_settings)
        self._builders[Strings.SIMULATION_BUILDER] = SimulationBuilder(self._settings.simulation_settings)

    def _build(self):
        self._builders[Strings.BIOME_BUILDER].build()
        self._builders[Strings.SIMULATION_BUILDER].build()

    def _build_context(self):
        try:
            self._setup_builders()
            self._build()
            self._context.set(Strings.BIOME_CONTEXT, self._builders[Strings.BIOME_BUILDER].context)
            self._context.set(Strings.SIMULATION_CONTEXT, self._builders[Strings.SIMULATION_BUILDER].context)
            if self._context is None:
                raise BootstrapError("[CRITICAL] Null context, aborting bootstrap.")
        except (BootstrapError, MapGenerationError, TypeError) as e:
            tb = traceback.format_exc()
            self._logger.exception(f"Error creating Context. Traceback: {tb}")
            self._logger.critical(f"[CRITICAL] Error while building the context: {e}")
            # sys.exit(1)

    def get_context(self) -> Context:
        return self._context
