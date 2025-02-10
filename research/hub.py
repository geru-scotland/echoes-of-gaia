from logging import Logger

from config.settings import Settings
from simulation.api.simulation_api import SimulationAPI
from simulation.render.manager import RenderManager
from utils.loggers import setup_logger

settings = Settings()
HEADLESS: bool = False
if not HEADLESS:
    render_manager = RenderManager(settings=settings.render_settings)
simulation_api: SimulationAPI = SimulationAPI(settings)
simulation_api.initialise()
logger: Logger = setup_logger("research", "research.log")
logger.info("Welcome too the Research Hub")
simulation_api.run()