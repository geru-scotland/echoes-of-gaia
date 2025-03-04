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

BiomeStore.load_ecosystem_data()


def test_prediction():
    print("\n=== TESTING PREDICTION ===\n")
    BiomeStore.load_ecosystem_data()

    from shared.enums import WeatherEvent, BiomeType, Season

    agent = ReinforcementLearningAgent(Agents.Reinforcement.NAIVE_CLIMATE)

    agent.load_model()


    test_observations = [
        {
            "temperature": np.array([0.65], dtype=np.float32),
            "atm_pressure": np.array([0.50], dtype=np.float32),
            "biome_type": 0,
            "season": 0
        },
        {
            "temperature": np.array([0.95], dtype=np.float32),
            "atm_pressure": np.array([0.30], dtype=np.float32),
            "biome_type": 1,
            "season": 1
        },
        {
            "temperature": np.array([0.10], dtype=np.float32),
            "atm_pressure": np.array([0.70], dtype=np.float32),
            "biome_type": 2,
            "season": 3
        },
        {
            "temperature": np.array([0.50], dtype=np.float32),
            "atm_pressure": np.array([0.40], dtype=np.float32),
            "biome_type": 3,
            "season": 2
        },
        {
            "temperature": np.array([0.40], dtype=np.float32),
            "atm_pressure": np.array([0.20], dtype=np.float32),
            "biome_type": 1,
            "season": 3
        },
        {
            "temperature": np.array([0.80], dtype=np.float32),
            "atm_pressure": np.array([0.60], dtype=np.float32),
            "biome_type": 0,
            "season": 1
        },
        {
            "temperature": np.array([0.50], dtype=np.float32),
            "atm_pressure": np.array([0.90], dtype=np.float32),
            "biome_type": 2,
            "season": 1
        },
        {
            "temperature": np.array([0.30], dtype=np.float32),
            "atm_pressure": np.array([0.80], dtype=np.float32),
            "biome_type": 0,
            "season": 3
        }
    ]

    def denormalize_temp(norm_temp):
        return norm_temp * 80 - 30

    for i, obs in enumerate(test_observations):
        print(f"\nObservación {i + 1}:")
        real_temp = denormalize_temp(obs['temperature'][0])
        print(f"  Temperatura: {obs['temperature'][0]:.2f} (normalizada) = {real_temp:.1f}°C")
        print(f"  Presión: {obs['atm_pressure'][0]:.2f} (normalizada)")
        print(f"  Bioma: {list(BiomeType)[obs['biome_type']]}")
        print(f"  Estación: {list(Season)[obs['season']]}")

        action = agent.predict(obs)
        if action < len(WeatherEvent):
            print(f"  Acción: {action} (WeatherEvent: {list(WeatherEvent)[action]})")
        else:
            print(f"  Acción: {action} (WeatherEvent: DESCONOCIDO)")
        print()


if __name__ == "__main__":
    # Para entrenar el modelo
    # BiomeStore.load_ecosystem_data()
    # train_agent = ReinforcementTrainingAgent(Agents.Reinforcement.NAIVE_CLIMATE)
    # train_agent.train()

    test_prediction()
    # train_agent: ReinforcementTrainingAgent = ReinforcementTrainingAgent(Agents.Reinforcement.NAIVE_CLIMATE)
    # train_agent.train()
