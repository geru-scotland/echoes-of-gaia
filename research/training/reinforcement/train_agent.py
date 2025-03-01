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
from typing import Type

# No borrar este import, es necesario para el EnvironmentRegistry.
import research.training.reinforcement.environments

from research.training.registry import EnvironmentRegistry
from shared.enums import Agents
from shared.strings import Loggers
from utils.loggers import LoggerManager


class ReinforcementTrainingAgent:
    def __init__(self, agent_type: Agents.Reinforcement):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        try:
            environment_class_name: Type = EnvironmentRegistry.get(agent_type)
            print(environment_class_name)
            self._model = environment_class_name()
        except Exception as e:
            self._logger.exception(f"Error when getting {agent_type} from the Environent Registry: {e}")


    def load_model(self) -> None:
        pass

    def save_model(self) -> None:
        pass

    def train(self) -> None:
        self._logger.info(f"Training...{self._model}")

    def predict(self) -> None:
        pass




