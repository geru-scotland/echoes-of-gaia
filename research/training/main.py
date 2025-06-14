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

"""
Entry point for training reinforcement learning agents in biome simulations.

Initializes the training environment with ecosystem data and instantiates
appropriate agents based on the selected type. Serves as the primary interface
for launching agent training sessions - handles environment preparation
and training process management.
"""

from research.training.reinforcement.fauna.fauna_env import FaunaEnvironment
from research.training.reinforcement.climate.naive_climate import NaiveClimateEnvironment
from research.training.reinforcement.training_agent import ReinforcementLearningAgent
from shared.enums.enums import Agents
from shared.stores.biome_store import BiomeStore

BiomeStore.load_ecosystem_data()

if __name__ == "__main__":
    # Para entrenar el modelo
    # BiomeStore.load_ecosystem_data()
    train_agent = ReinforcementLearningAgent(Agents.Reinforcement.FAUNA)
    train_agent.train()
