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
from research.training.reinforcement.rl_model import ReinforcementModel
from shared.enums.strings import Loggers
from shared.types import Observation
from shared.enums.enums import ActionType, FaunaAction
from biome.agents.base import Agent
from utils.loggers import LoggerManager


class FaunaAgentAI(Agent[Observation, FaunaAction]):
    def __init__(self, fauna_model: str):
        self._model = ReinforcementModel(fauna_model)
        self._logger = LoggerManager.get_logger(Loggers.FAUNA_AGENT)

    def perceive(self) -> Observation:
        pass

    def decide(self, observation: Observation) -> FaunaAction:
        pass

    def act(self, action: FaunaAction) -> None:
        pass