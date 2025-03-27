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

from stable_baselines3 import PPO, DQN, A2C, SAC
from .config_loader import ConfigLoader
from shared.types import Observation


class ReinforcementModel:
    ALGORITHMS = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
        "SAC": SAC
    }

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model_name = os.path.basename(model_path).split('_')[0]
        self._config = ConfigLoader().get_config(self._model_name)

        algorithm = self._config["model"]["algorithm"]
        self._model = self.ALGORITHMS[algorithm].load(model_path)

    def predict(self, observation: Observation) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return action
