from logging import Logger
from threading import Thread

from config.settings import Settings
from simulation.api.simulation_api import SimulationAPI
from simulation.render.manager import RenderManager
from utils.loggers import setup_logger

settings = Settings()
# TODO: Pasar a config esto
HEADLESS: bool = True

simulation = SimulationAPI(settings)

logger = setup_logger("research", "research.log")
logger.info("Welcome to the Research Hub")


if HEADLESS:
    simulation.run()
else:
    sim_thread = Thread(target=simulation.run, daemon=True)
    sim_thread.start()
    # Es bloqueante del hilo principal, lo pongo al final
    render_manager = RenderManager(settings=settings.render_settings)
    render_manager.start_engine()
