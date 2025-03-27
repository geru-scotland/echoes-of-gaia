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
from logging import Logger

from stable_baselines3 import PPO, DQN, A2C, SAC

from shared.enums.enums import Agents
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager
from utils.paths import get_model_path
from .config_loader import ConfigLoader
from shared.types import Observation


class ReinforcementLearningModel:
    ALGORITHMS = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
        "SAC": SAC
    }

    def __init__(self, agent_type: Agents.Reinforcement, model_path: str):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._model_path = get_model_path(model_path)
        self._config = ConfigLoader().get_config(agent_type)

        try:
            self._logger.info(f"Loading Reinforcement model from {self._model_path}...")
            algorithm = self._config["model"]["algorithm"]
            self._model = self.ALGORITHMS[algorithm].load(self._model_path)
            self._logger.info(f"Model loaded! Using {algorithm} algorithm.")
        except Exception as e:
            self._logger.exception(f"Error loading model: {e}")

    def predict(self, observation: Observation) -> int:
        action, _ = self._model.predict(observation, deterministic=True)
        return action
