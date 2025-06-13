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
import os
import random
from typing import Any, Dict

import yaml

from utils.paths import CONFIG_DIR


class TrainingConfigManager:

    @staticmethod
    def generate_random_config(base_config_file: str) -> Dict[str, Any]:
        filepath = os.path.join(CONFIG_DIR, base_config_file)
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)

        biome_config = TrainingConfigManager._load_random_biome_config()

        if 'biome' in config and 'biome' in biome_config:
            config['biome']['type'] = biome_config['biome']['type']
            config['biome']['flora'] = biome_config['biome']['flora']
            config['biome']['fauna'] = biome_config['biome']['fauna']
            config['biome']['available_fauna'] = [entry['species'] for entry in biome_config['biome']['fauna']]
            config['biome']['available_flora'] = [entry['species'] for entry in biome_config['biome']['flora']]

        config = TrainingConfigManager._modify_config_if_needed(config)

        return config

    @staticmethod
    def _load_random_biome_config() -> Dict[str, Any]:
        biome_configs_dir = os.path.join(CONFIG_DIR, "biomes")

        if not os.path.exists(biome_configs_dir):
            raise FileNotFoundError(f"The directory doesn't exist: {biome_configs_dir}")

        yaml_files = [f for f in os.listdir(biome_configs_dir)
                      if f.endswith('.yaml') and not f.startswith('_')]

        if not yaml_files:
            raise ValueError(f"There aren't any YAML file with folder: {biome_configs_dir}")

        random_biome_file = random.choice(yaml_files)

        biome_config_path = os.path.join(biome_configs_dir, random_biome_file)

        with open(biome_config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def _modify_config_if_needed(config: Dict[str, Any]) -> Dict[str, Any]:
        # Pongo esto de ejemplo, cambiar cuando quiera más variabilidad.
        if 'biome' in config and 'fauna' in config['biome']:
            for fauna in config['biome']['fauna']:
                if fauna.get('species') == 'deer':
                    fauna['spawns'] = random.randint(1, 7)

        return config

    @staticmethod
    def save_temp_config(config: Dict[str, Any], custom_name: str = None, output_path: str = None) -> str:
        if output_path and custom_name:
            temp_path = os.path.join(output_path, custom_name)
        else:
            temp_path = os.path.join(os.path.dirname(__file__), 'temp_training_config.yaml')

        with open(temp_path, 'w') as file:
            yaml.dump(config, file)
        return temp_path
