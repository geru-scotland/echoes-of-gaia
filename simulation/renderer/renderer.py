from simulation.core.systems.events.dispatcher import EventDispatcher


class Renderer:
    def __init__(self):
        EventDispatcher.register("biome_loaded", self.on_biome_loaded)

    def on_biome_loaded(self):
        print("[Render] Biome Loaded! Now biome image should be projected")