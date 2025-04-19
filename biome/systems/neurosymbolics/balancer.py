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
from typing import Dict, Type, Optional

from biome.services.biome_control_service import BiomeControlService
from biome.systems.neurosymbolics.data_service import NeurosymbolicDataService
from biome.systems.neurosymbolics.integrations.base_strategy import IntegrationStrategy
from biome.systems.neurosymbolics.modules.neural_module import NeuralModuleInterface
from biome.systems.neurosymbolics.modules.symbolic_module import SymbolicModuleInterface
from shared.enums.strings import Loggers
from shared.types import Observation, IntegratedResult, PredictionFeedback, SymbolicFeedback
from utils.loggers import LoggerManager


class NeuroSymbolicBalancer:
    def __init__(self,
                 neural_module: Type[NeuralModuleInterface],
                 symbolic_module: Type[SymbolicModuleInterface],
                 integration_strategy: Type[IntegrationStrategy]):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initialising NeurySymbolic balancer...")
        self.neural_module: NeuralModuleInterface = neural_module()
        self.symbolic_module: SymbolicModuleInterface = symbolic_module()
        self.integration_strategy: IntegrationStrategy = integration_strategy()
        self.confidence_weights: Dict[str, float] = {"neural": 0.6, "symbolic": 0.4}

    def get_observation(self, data_service: NeurosymbolicDataService) -> Observation:
        neural_sequence = data_service.get_neural_sequence()
        graph_data = data_service.get_graph_data()

        return {
            'neural_data': neural_sequence,
            'graph_data': graph_data
        }

    def process(self, observation: Observation) -> Optional[IntegratedResult]:
        neural_feedback: PredictionFeedback = self.neural_module.predict(observation.get("neural_data", {}))
        symbolic_feedback: SymbolicFeedback = self.symbolic_module.infer(observation.get("graph_data", {}))
        self._logger.debug(f"[NeuroSymbolic Balancer] Neural feedback: {neural_feedback}")
        self._logger.debug(f"[NeuroSymbolic Balancer] Symbolic feedback: {symbolic_feedback}")
        #
        # integrated_result = self.integration_strategy.integrate(
        #     neural_feedback,
        #     symbolic_feedback,
        #     self.confidence_weights
        # )

        return neural_feedback
