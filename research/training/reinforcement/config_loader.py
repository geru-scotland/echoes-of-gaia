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
import yaml
from typing import Dict, Optional

from shared.enums.enums import ReinforcementConfig


class ConfigLoader:
    _instance = None
    _configs: Dict[str, ReinforcementConfig] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load_configs()
        return cls._instance

    def _load_configs(self) -> None:
        config_dir = os.path.join(os.path.dirname(__file__), 'config')
        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml'):
                model_name = os.path.splitext(filename)[0]
                with open(os.path.join(config_dir, filename), 'r') as f:
                    self._configs[model_name] = yaml.safe_load(f)

    def get_config(self, model_name: str) -> Optional[ReinforcementConfig]:
        return self._configs.get(model_name)
