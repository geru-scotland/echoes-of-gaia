from logging import Logger
from typing import Dict, Any

from biome.systems.maps.procedural_maps import Map
from simulation.core.systems.events.dispatcher import EventDispatcher
from simulation.core.systems.events.handler import EventHandler
from simulation.render.components import MapComponent
from simulation.render.engine import RenderEngine


class RenderEventHandler(EventHandler):
    def __init__(self, engine: RenderEngine):
        super().__init__()
        self._engine = engine
        self._settings = engine.settings
        self._logger: Logger = self._settings.get_logger("render")

    def _register_events(self):
        EventDispatcher.register("biome_loaded", self.on_biome_loaded)

    def on_biome_loaded(self, map: Map):
        print("[Render] Biome Loaded! Now biome image should be projected")
        print(f"Render title: {self._settings.title}")
        try:
            if not self._engine.is_initialized():
                self._engine.init()
            tile_config: Dict[str, Any] = self._settings.config.get("tiles", {})
            map_component: MapComponent = MapComponent(map, self._settings.window_width,
                                                       self._settings.window_height,
                                                       tile_config)
            self._engine.add_component(map_component)
        except Exception as e:
            self._logger.exception(f"There was an error adding the Map component to the Render Engine: {e}")
