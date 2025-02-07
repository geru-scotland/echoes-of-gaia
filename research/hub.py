from logging import Logger

from simulation.api.simulation_api import SimulationAPI
from simulation.renderer.renderer import Renderer
from utils.loggers import setup_logger

simulation_api: SimulationAPI = SimulationAPI(render=Renderer)
simulation_api.initialise()
logger: Logger = setup_logger("research", "research.log")
logger.info("Welcome too the Research Hub")
simulation_api.run()