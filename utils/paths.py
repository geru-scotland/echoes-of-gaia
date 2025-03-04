"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BIOME_DIR = os.path.join(BASE_DIR, 'biome')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
GAME_DIR = os.path.join(BASE_DIR, 'game')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
SIMULATION_DIR = os.path.join(BASE_DIR, 'simulation')
RESEARCH_DIR = os.path.join(BASE_DIR, 'research')
ASSETS_DIR = os.path.join(GAME_DIR, 'assets')
UTILS_DIR = os.path.join(GAME_DIR, 'utils')
BIOME_DATA_DIR = os.path.join(BIOME_DIR, 'data')

TRAINING_DIR = os.path.join(RESEARCH_DIR, 'training')
MODELS_DIR = os.path.join(TRAINING_DIR, 'models')

def get_model_path(model_name: str) -> str:
    return os.path.join(MODELS_DIR, model_name)