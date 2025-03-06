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
import numpy as np

from research.training.reinforcement.train_agent import ReinforcementLearningAgent
from shared.enums import Agents, BiomeType, Season, WeatherEvent
from shared.stores.biome_store import BiomeStore
from utils.normalization.normalizer import climate_normalizer

BiomeStore.load_ecosystem_data()



if __name__ == "__main__":
    # Para entrenar el modelo
    # BiomeStore.load_ecosystem_data()
    train_agent = ReinforcementLearningAgent(Agents.Reinforcement.NAIVE_CLIMATE)
    train_agent.train()

    # train_agent: ReinforcementTrainingAgent = ReinforcementTrainingAgent(Agents.Reinforcement.NAIVE_CLIMATE)
    # train_agent.train()
