from logging import Logger

from simulation.api.simulation_api import SimulationAPI
from utils.loggers import setup_logger

simulation_api: SimulationAPI = SimulationAPI()
simulation_api.initialise()
logger: Logger = setup_logger("research", "research.log")
logger.info("Welcome too the Research Hub")
simulation_api.run()