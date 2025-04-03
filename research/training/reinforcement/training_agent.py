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
import traceback
from typing import Type
from logging import Logger

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from research.training.registry import EnvironmentRegistry

import stable_baselines3 as sb3
from stable_baselines3 import PPO, DQN, A2C, SAC

from research.training.reinforcement.config_loader import ConfigLoader
from research.training.reinforcement.fauna.cnn_feature_extractor import create_custom_cnn_policy
from shared.enums.enums import Agents, LocalFovConfig
from utils.loggers import LoggerManager
from shared.enums.strings import Loggers


class SaveEmbeddingsCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def __init__(self, save_path="./embeddings", verbose=0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_training_end(self) -> None:
        extractor = None

        # Intento primero para PPO/A2C
        if hasattr(self.model, "policy"):
            extractor = getattr(self.model.policy, "features_extractor", None)

        # Si no lo encuentra y es DQN, entonces q_net
        if extractor is None and hasattr(self.model, "q_net"):
            extractor = getattr(self.model.q_net, "features_extractor", None)

        if extractor and hasattr(extractor, "save_terrain_embeddings"):
            extractor.save_terrain_embeddings(self.save_path)
        else:
            print("[WARN] No extractor or method save_terrain_embeddings found.")


class ReinforcementLearningAgent:
    ALGORITHMS = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
        "SAC": SAC
    }

    def __init__(self, agent_type: Agents.Reinforcement):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._model_name = agent_type
        self._config = ConfigLoader().get_config(agent_type)

        if not self._config:
            raise ValueError(f"No configuration found for model: {agent_type}")

        env_class_name = self._config["environment"]["env_class"]
        try:
            environment_class: Type = EnvironmentRegistry.get_by_name(env_class_name)
            local_fov_config: LocalFovConfig = self._config["local_fov"]
            self._environment: gym.Env = environment_class(local_fov_config)
        except Exception as e:
            self._logger.exception(f"Error retrieving environment {agent_type}: {e}")
            raise

    def train(self) -> None:
        self._logger.info(f"Starting training with algorithm {self._config['model']['algorithm']}...")
        self._logger.info(f"Gymnasium version: {gym.__version__}")
        self._logger.info(f"SB3 version: {sb3.__version__}")

        try:
            algorithm_class = self.ALGORITHMS.get(self._config["model"]["algorithm"])
            if not algorithm_class:
                raise ValueError(f"Unsupported algorithm: {self._config['model']['algorithm']}")

            policy_kwargs = create_custom_cnn_policy()
            callback = SaveEmbeddingsCallback()
            sb3_model = algorithm_class(
                policy=self._config["model"]["policy"],
                policy_kwargs=policy_kwargs,
                env=self._environment,
                **self._config["model"]["hyperparams"]
            )

            total_timesteps = self._config["model"]["timesteps"]
            sb3_model.learn(total_timesteps=total_timesteps, callback=callback)

            output_path = self._config["output_path"]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sb3_model.save(output_path)

            self._logger.info(f"Training completed after {total_timesteps} timesteps!")

        except Exception as e:
            self._logger.exception(f"Error during training: {e}", exc_info=True)
            traceback.print_exc()
            raise
