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
import random
from typing import Dict, Any, List, Optional
import yaml
import os

from shared.enums.enums import BiomeType, FloraSpecies, FaunaSpecies
from utils.paths import CONFIG_DIR


class TrainingConfigManager:

    @staticmethod
    def generate_random_config(base_config_file: str) -> Dict[str, Any]:
        filepath = os.path.join(CONFIG_DIR, base_config_file)
        with open(filepath, 'r') as file:
            print(filepath)
            config = yaml.safe_load(file)

        # random_biome = random.choice(list(BiomeType))
        # config['biome']['type'] = str(random_biome).lower()

        config['biome']['fauna'] = [
            {
                "species": str(FaunaSpecies.DEER).lower(),
                "spawns": 1,
                "avg-lifespan": random.randint(1, 7),
                "components": [
                    {"GrowthComponent": {}},
                    {"VitalComponent": {}},
                    {"WeatherAdaptationComponent": {
                        "optimal_temperature": random.uniform(5.0, 30.0)
                    }},
                    {"MovementComponent": {}},
                    {"HeterotrophicNutritionComponent": {}}
                ]
            }
        ]

        return config

    @staticmethod
    def save_temp_config(config: Dict[str, Any]) -> str:
        temp_path = os.path.join(os.path.dirname(__file__), 'temp_training_config.yaml')
        with open(temp_path, 'w') as file:
            yaml.dump(config, file)
        return temp_path
