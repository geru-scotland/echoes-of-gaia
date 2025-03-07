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
from typing import Optional, Dict, Any

from stable_baselines3 import PPO

from shared.enums.strings import Loggers
from utils.loggers import LoggerManager
from utils.normalization.normalizer import climate_normalizer
from utils.paths import get_model_path


class ReinforcementModel:
    def __init__(self, model_name: str = "climate_model_final"):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._model_name: str = model_name
        self._model = None

    def _load_model(self) -> None:
        try:
            model_path = get_model_path(self._model_name)
            self._logger.info(f"Loading Reinforcement model from {model_path}...")
            self._model = PPO.load(model_path, env=self._model)
            self._logger.info(f"Model loaded!")
        except Exception as e:
            self._logger.exception(f"Error loading model: {e}")

    def predict(self, observation: Optional[Dict[str, Any]] = None) -> int:
        from shared.enums.enums import WeatherEvent

        if  self._model is None:
            self._load_model()
            if self._model is None:
                self._logger.error("The model coudln't be loaded.")
                return 0

        try:
            if observation is None:
                observation, _ = self._model.reset()
                self._logger.info(f"Using reset's observation as no argument was given: {observation}")

            action, _states = self._model.predict(observation, deterministic=True)

            temp_normalized = observation["temperature"][0]
            temp_real = climate_normalizer.denormalize("temperature", temp_normalized)
            humidity_normalized = observation["humidity"][0]
            humidity_real = climate_normalizer.denormalize("humidity",  humidity_normalized)
            precipitation_normlaized= observation["precipitation"][0]
            precipitation_real = climate_normalizer.denormalize("precipitation", precipitation_normlaized)

            weather_event_name = list(WeatherEvent)[action] if action < len(WeatherEvent) else "UNKNOWN"

            self._logger.debug(f"Real temperature: {temp_real:.1f}°C")
            self._logger.debug(f"Real humidity: {humidity_real:.1f}°C")
            self._logger.debug(f"Real precipitation: {precipitation_real:.1f}°C")
            self._logger.debug(f"Predicted action: {action} (Weather Event: {weather_event_name})")
            return action
        except Exception as e:
            self._logger.exception(f"There was an error during the prediction: {e}")
            return 0