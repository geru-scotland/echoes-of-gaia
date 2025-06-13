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
Bridges simulation events with render system operations.

Registers handlers for biome loading and simulation completion events;
creates map components and integrates them with render engine.
Manages render lifecycle in response to simulation state changes.
"""

from logging import Logger
from typing import Any, Dict

from shared.enums.strings import Loggers
from shared.events.handler import EventHandler
from shared.types import TileMap
from simulation.core.systems.events.event_bus import SimulationEventBus
from simulation.render.components import MapComponent
from simulation.render.engine import RenderEngine
from utils.loggers import LoggerManager


class RenderEventHandler(EventHandler):
    def __init__(self, engine: RenderEngine):
        super().__init__()
        self._engine = engine
        self._settings = engine.settings
        self._logger: Logger = LoggerManager.get_logger(Loggers.RENDER)

    def _register_events(self):
        SimulationEventBus.register("biome_loaded", self.on_biome_loaded)
        SimulationEventBus.register("simulation_finished", self.on_simulation_finished)

    def on_biome_loaded(self, tile_map: TileMap):
        self._logger.info("[Render] Biome Loaded! Now biome image should be projected")
        self._logger.info(f"Render title: {self._settings.title}")
        try:
            if self._engine.is_initialized():
                tile_config: Dict[str, Any] = self._settings.config.get("tiles", {})
                map_component: MapComponent = MapComponent(tile_map, tile_config)
                self._engine.enqueue_task(self._engine.add_component, map_component)
        except Exception as e:
            self._logger.exception(f"There was an error adding the Map component to the Render Engine: {e}")

    def on_simulation_finished(self):
        self._logger.info("Finishing simulation")