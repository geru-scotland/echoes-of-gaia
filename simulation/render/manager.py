from config.settings import RenderSettings
from simulation.render.engine import RenderEngine
from simulation.render.event_handler import RenderEventHandler

class RenderManager:
    def __init__(self, settings: RenderSettings):
        self._settings = settings
        self._engine = RenderEngine(settings)
        self._event_handler = RenderEventHandler(self._engine)

    def start_engine(self):
        self._engine.init()

    @property
    def engine(self) -> RenderEngine:
        return self._engine
