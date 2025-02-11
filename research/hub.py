from logging import Logger
from threading import Thread

from config.settings import Settings
from simulation.api.simulation_api import SimulationAPI
from simulation.render.manager import RenderManager
from utils.loggers import setup_logger

settings = Settings()
# TODO: Pasar a config esto
HEADLESS: bool = False

simulation_api: SimulationAPI = SimulationAPI(settings)
# simulation_api .initialise()
sim_thread = Thread(target=simulation_api.run, daemon=True)
sim_thread.start()

logger: Logger = setup_logger("research", "research.log")
logger.info("Welcome too the Research Hub")

if not HEADLESS:
    # Es bloqueante del hilo principal, lo pongo al final
    render_manager = RenderManager(settings=settings.render_settings)
    render_manager.start_engine()
