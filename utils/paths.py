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
DATASET_DIR = os.path.join(SIMULATION_DIR, 'dataset')
RESEARCH_DIR = os.path.join(BASE_DIR, 'research')
ASSETS_DIR = os.path.join(GAME_DIR, 'assets')
UTILS_DIR = os.path.join(GAME_DIR, 'utils')
BIOME_DATA_DIR = os.path.join(BIOME_DIR, 'data')
DATASET_GENERATED_CONFIGS_DIR = os.path.join(DATASET_DIR, 'generated_configs')

TRAINING_DIR = os.path.join(RESEARCH_DIR, 'training')
MODELS_DIR = os.path.join(TRAINING_DIR, 'models')

NEUROSYBOLIC_DATA_DIR = os.path.join(RESEARCH_DIR, 'data', 'neurosymbolic')
DEEP_LEARNING_DIR = os.path.join(TRAINING_DIR, 'deep_learning')
DEEP_LEARNING_CONFIG_DIR = os.path.join(DEEP_LEARNING_DIR, 'config')

REINFORCEMENT_DIR = os.path.join(TRAINING_DIR, 'reinforcement')
REINFORCEMENT_CONFIG_DIR = os.path.join(REINFORCEMENT_DIR, 'config')


def get_model_path(model_name: str) -> str:
    return os.path.join(MODELS_DIR, model_name)
