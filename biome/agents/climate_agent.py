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

import numpy as np

from biome.agents.base import Agent
from biome.systems.climate.state import ClimateState
from biome.systems.climate.system import ClimateSystem
from research.training.reinforcement.rl_model import ReinforcementModel
from shared.enums.enums import WeatherEvent, BiomeType, Season
from shared.enums.strings import Loggers
from shared.types import Observation
from utils.loggers import LoggerManager
from shared.normalization.normalizer import climate_normalizer


class ClimateAgentAI(Agent[ClimateState, WeatherEvent]):
    def __init__(self, climate: ClimateSystem, climate_model: str):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE_AGENT)
        self._climate: ClimateSystem = climate
        self._model: ReinforcementModel = ReinforcementModel(climate_model)

    def perceive(self) -> Observation:
        self._logger.debug("AGENT PERCEIVING")
        biome_idx: int = list(BiomeType).index(self._climate.biome_type)
        season_idx: int = list(Season).index(self._climate.get_current_season())
        # TODO: implementar get_state_with_modifiers() cuando llegue el momento
        # para aplicar al clima modificadores
        state: ClimateState = self._climate.get_state_and_record()

        normalized_temp: float = climate_normalizer.normalize("temperature", state.temperature)
        normalized_humidity: float = climate_normalizer.normalize("humidity", state.humidity)
        normalized_precipitation: float = climate_normalizer.normalize("precipitation", state.precipitation)

        return {
            "temperature": np.array([normalized_temp], dtype=np.float32),
            "humidity": np.array([normalized_humidity], dtype=np.float32),
            "precipitation": np.array([normalized_precipitation], dtype=np.float32),
            "biome_type": biome_idx,
            "season": season_idx
        }

    def decide(self, observation: Observation) -> WeatherEvent:
        self._logger.debug(f"AGENT DECIDING. Observation: {observation}")
        weather_event_idx: int = self._model.predict(observation)
        return WeatherEvent(list(WeatherEvent)[weather_event_idx])

    def act(self, action: WeatherEvent) -> None:
        self._logger.debug(f"AGENT ACTING. Action: {action}")
        self._climate.update(action)

