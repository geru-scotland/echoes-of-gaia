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


def test_prediction():
    print("\n=== TESTING PREDICTION ===\n")
    BiomeStore.load_ecosystem_data()

    from shared.enums import WeatherEvent, BiomeType, Season

    agent = ReinforcementLearningAgent(Agents.Reinforcement.NAIVE_CLIMATE)

    agent.load_model()

    test_observations_raw = [
        {
            "temperature": 22.0,
            "atm_pressure": 1000.0,
            "biome_type": 0,
            "season": 0
        },
        {
            "temperature": 42.0,
            "atm_pressure": 980.0,
            "biome_type": 1,
            "season": 1
        },
        {
            "temperature": -14.0,
            "atm_pressure": 1020.0,
            "biome_type": 2,
            "season": 3
        },
        {
            "temperature": 10.0,
            "atm_pressure": 990.0,
            "biome_type": 3,
            "season": 2
        },
        {
            "temperature": 5.0,
            "atm_pressure": 970.0,
            "biome_type": 1,
            "season": 3
        },
        {
            "temperature": 34.0,
            "atm_pressure": 1010.0,
            "biome_type": 0,
            "season": 1
        },
        {
            "temperature": 10.0,
            "atm_pressure": 1040.0,
            "biome_type": 2,
            "season": 1
        },
        {
            "temperature": 0.0,
            "atm_pressure": 1030.0,
            "biome_type": 0,
            "season": 3
        }
    ]

    test_observations = []
    for obs in test_observations_raw:
        normalized_obs = {
            "temperature": np.array([climate_normalizer.normalize('temperature', obs["temperature"])],
                                    dtype=np.float32),
            "atm_pressure": np.array([climate_normalizer.normalize('atm_pressure', obs["atm_pressure"])], dtype=np.float32),
            "biome_type": obs["biome_type"],
            "season": obs["season"]
        }
        test_observations.append(normalized_obs)


    for i, obs in enumerate(test_observations):
        print(f"\nObservación {i + 1}:")
        real_temp = climate_normalizer.denormalize("temperature", obs['temperature'][0])
        real_atm_pressure = climate_normalizer.denormalize("atm_pressure", obs['atm_pressure'][0])
        print(f"  Temperatura: {obs['temperature'][0]:.2f} (normalizada) = {real_temp:.1f}°C")
        print(f"  Presión: {obs['atm_pressure'][0]:.2f} (normalizada) = {real_atm_pressure} hPa")
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
