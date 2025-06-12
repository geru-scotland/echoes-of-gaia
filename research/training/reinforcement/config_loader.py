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
Configuration loader for reinforcement learning training components.

Manages the loading and access of YAML configuration files for agent training;
implements a singleton pattern to ensure consistent configuration access.
Handles both agent-specific parameters and global biome settings - provides
unified access to training hyperparameters and environment configurations.
"""

import os
from pathlib import Path

import yaml
from typing import Dict, Optional, Any

from shared.enums.enums import ReinforcementConfig
from utils.paths import CONFIG_DIR


class ConfigLoader:
    _instance = None
    _agent_config: Dict[str, ReinforcementConfig] = {}
    _biome_config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self) -> None:

        # Parte de reinforcment
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml'):
                model_name = os.path.splitext(filename)[0]
                with open(os.path.join(config_dir, filename), 'r') as f:
                    self._agent_config[model_name] = yaml.safe_load(f)

        # Y ahora le inyecto el fov, que lo he llevado a biome.yaml
        biome_config_path: Path = Path(os.path.join(CONFIG_DIR), "training.yaml")
        with open(biome_config_path, 'r') as f:
            biome_configs = yaml.safe_load(f)
            local_fov: Dict[str, Any] = biome_configs.get("biome", {}).get("map", {}).get("local_fov", {})
            self._biome_config.update({"local_fov": local_fov})

    def get_agent_config(self, model_name: str) -> Optional[ReinforcementConfig]:
        return self._agent_config.get(model_name)

    def get_biome_config(self) -> Dict[str, Any]:
        return self._biome_config
