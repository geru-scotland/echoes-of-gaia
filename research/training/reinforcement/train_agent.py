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
import traceback
from logging import Logger
from typing import Type

import stable_baselines3

# No borrar este import, es necesario para el EnvironmentRegistry.

from stable_baselines3 import PPO
import gymnasium as gym

from research.training.registry import EnvironmentRegistry
from shared.enums.enums import Agents
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class ReinforcementLearningAgent:
    def __init__(self, agent_type: Agents.Reinforcement):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        try:
            environment_class_name: Type = EnvironmentRegistry.get(agent_type)
            self._environment: gym.Env = environment_class_name()
        except Exception as e:
            self._logger.exception(f"Error when getting {agent_type} from the Environent Registry: {e}")

    def train(self) -> None:
        self._logger.info(f"Starting training with PPO algorithm...")
        self._logger.info(f"Gymnasium version: {gym.__version__}")
        self._logger.info(f"SB3 version: {stable_baselines3.__version__}")

        try:
            sb3_model = PPO(
                policy="MultiInputPolicy",
                env=self._environment,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef= 0.02, # Coef entropia, este valor incentiva expl, pero mantiene aleatoriedad.
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )

            total_timesteps = 1000000
            sb3_model.learn(
                total_timesteps=total_timesteps
            )

            # TODO: Mejorar paths
            sb3_model.save("./models/climate_model_final")

            self._logger.info(f"Training completed after {total_timesteps} timesteps!")

        except Exception as e:
            self._logger.exception(f"Error during training: {e}", exc_info=True)
            traceback.print_exc()

