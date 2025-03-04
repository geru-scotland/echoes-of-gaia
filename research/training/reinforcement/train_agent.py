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
from logging import Logger
from typing import Type, Optional, Dict, Any

import stable_baselines3

# No borrar este import, es necesario para el EnvironmentRegistry.
import research.training.reinforcement.environments

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

from research.training.registry import EnvironmentRegistry
from shared.enums import Agents
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ReinforcementLearningAgent:
    def __init__(self, agent_type: Agents.Reinforcement):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        try:
            environment_class_name: Type = EnvironmentRegistry.get(agent_type)
            self._environment: gym.Env = environment_class_name()
            self._model = None
        except Exception as e:
            self._logger.exception(f"Error when getting {agent_type} from the Environent Registry: {e}")


    def save_model(self) -> None:
        pass

    def train(self) -> None:
        self._logger.info(f"Starting training with PPO algorithm...")
        self._logger.info(f"Gymnasium version: {gym.__version__}")
        self._logger.info(f"SB3 version: {stable_baselines3.__version__}")

        try:
            model = PPO(
                policy="MultiInputPolicy",
                env=self._environment,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1,
                tensorboard_log="./tensorboard_logs/"
            )

            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path="./models/",
                name_prefix="climate_model"
            )

            total_timesteps = 100000
            model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_callback
            )

            model.save("./models/climate_model_final")
            self._model = model

            self._logger.info(f"Training completed after {total_timesteps} timesteps!")

        except Exception as e:
            self._logger.exception(f"Error during training: {e}")

    def load_model(self, path: str = "./models/climate_model_final") -> None:
        try:
            from stable_baselines3 import PPO
            self._model = PPO.load(path, env=self._model)
            self._logger.info(f"Model loaded from {path}")
        except Exception as e:
            self._logger.exception(f"Error loading model: {e}")

    def predict(self, observation: Optional[Dict[str, Any]] = None) -> int:
        from shared.enums import WeatherEvent

        if  self._model is None:
            self.load_model()
            if self._model is None:
                self._logger.error("The model coudln't be loaded.")
                return 0

        try:
            if observation is None:
                observation, _ = self._model.reset()
                self._logger.info(f"Using reset's observation as no argument was given: {observation}")

            action, _states = self._model.predict(observation, deterministic=True)

            temp_normalized = observation["temperature"][0]
            temp_real = temp_normalized * 80 - 30

            weather_event_name = list(WeatherEvent)[action] if action < len(WeatherEvent) else "UNKNOWN"

            self._logger.info(f"Real temperature: {temp_real:.1f}°C")
            self._logger.info(f"Predicted action: {action} (Weather Event: {weather_event_name})")
            return action
        except Exception as e:
            self._logger.exception(f"There was an error during the prediction: {e}")
            return 0




