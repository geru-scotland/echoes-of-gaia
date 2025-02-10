from biome.systems.maps.procedural_maps import Map
from simulation.core.systems.events.dispatcher import EventDispatcher
from simulation.render.components import MapComponent
from simulation.render.engine import RenderEngine


class RenderEventHandler:
    def __init__(self, engine: RenderEngine):
        self._engine = engine
        self._settings = engine.settings
        self._register_events()

    def _register_events(self):
        EventDispatcher.register("biome_loaded", self.on_biome_loaded)

    def on_biome_loaded(self, map: Map):
        print("[Render] Biome Loaded! Now biome image should be projected")
        print(f"Render title: {self._settings.title}")
        if not self._engine.is_initialized():
            self._engine.init()

        map_component: MapComponent = MapComponent(map, self._settings.window_width, self._settings.window_height)
        self._engine.add_component(map_component)
