""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""
import sys
from logging import Logger
from threading import Thread

from config.settings import Settings
from shared.enums.strings import Loggers
from simulation.api.simulation_api import SimulationAPI
from simulation.render.manager import RenderManager
from utils.loggers import LoggerManager
from exceptions.general import global_exception_handler

sys.excepthook = global_exception_handler

settings: Settings = Settings()
LoggerManager.initialize(settings.log_level)
# TODO: Pasar a config esto
HEADLESS: bool = True

simulation = SimulationAPI(settings)

logger: Logger = LoggerManager.get_logger(Loggers.RESEARCH)


if HEADLESS:
    simulation.run()
else:
    sim_thread = Thread(target=simulation.run, daemon=True)
    sim_thread.start()
    # Es bloqueante del hilo principal, lo pongo al final
    render_manager = RenderManager(settings=settings.render_settings)
    render_manager.start_engine()