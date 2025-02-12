from logging import Logger
from typing import Dict, Any

from biome.systems.maps.procedural_maps import Map
from shared.strings import Loggers
from simulation.core.systems.events.dispatcher import EventDispatcher
from simulation.core.systems.events.handler import EventHandler
from simulation.render.components import MapComponent
from simulation.render.engine import RenderEngine


class RenderEventHandler(EventHandler):
    def __init__(self, engine: RenderEngine):
        super().__init__()
        self._engine = engine
        self._settings = engine.settings
        self._logger: Logger = self._settings.get_logger(Loggers.RENDER)

    def _register_events(self):
        EventDispatcher.register("biome_loaded", self.on_biome_loaded)
        EventDispatcher.register("simulation_finished", self.on_simulation_finished)

    def on_biome_loaded(self, map: Map):
        print("[Render] Biome Loaded! Now biome image should be projected")
        print(f"Render title: {self._settings.title}")
        try:
            if self._engine.is_initialized():
                tile_config: Dict[str, Any] = self._settings.config.get("tiles", {})
                map_component: MapComponent = MapComponent(map, tile_config)
                self._engine.enqueue_task(self._engine.add_component, map_component)
        except Exception as e:
            self._logger.exception(f"There was an error adding the Map component to the Render Engine: {e}")

    def on_simulation_finished(self):
        self._logger.info("Finishing simulation")