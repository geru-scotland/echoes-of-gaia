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

from biome.agents.base import Agent, TAction, TState
from biome.systems.neurosymbolics.balancer import NeuroSymbolicBalancer
from biome.systems.neurosymbolics.data_service import NeurosymbolicDataService
from biome.systems.neurosymbolics.integrations.weighted_integration import WeightedIntegrationStrategy
from biome.systems.neurosymbolics.modules.neural_module import NeuralModule
from biome.systems.neurosymbolics.modules.symbolic_module import RuleBasedSymbolicModule
from shared.enums.strings import Loggers
from shared.types import Observation
from utils.loggers import LoggerManager


class EquilibriumAgentAI(Agent):

    def __init__(self):
        super().__init__()
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._data_service: NeurosymbolicDataService = NeurosymbolicDataService.get_instance()
        self._neurosymbolic_balancer: NeuroSymbolicBalancer = NeuroSymbolicBalancer(
            NeuralModule, RuleBasedSymbolicModule, WeightedIntegrationStrategy
        )

    def perceive(self) -> TState:
        self._logger.info(f"Equilibrium agent is perceiving...")
        observation: Observation = self._neurosymbolic_balancer.get_observation(self._data_service)
        return observation

    def decide(self, observation: TState) -> TAction:
        self._logger.info(f"Equilibrium agent is deceding...")
        neural_result = self._neurosymbolic_balancer.process(observation)
        self._data_service.clear_sequence_history()

    def act(self, action: TAction) -> None:
        # Lo llamo action por seguir y ser consistente, pero realmente es la
        # intervención en el bioma.
        self._logger.info(f"Equilibrium agent is acting...")
