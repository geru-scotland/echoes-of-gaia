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

from biome.systems.neurosymbolics.data_service import NeurosymbolicDataService
from biome.systems.neurosymbolics.integrations.base_strategy import IntegrationStrategy
from biome.systems.neurosymbolics.modules.neural_module import NeuralModuleInterface
from biome.systems.neurosymbolics.modules.symbolic_module import SymbolicModuleInterface
from shared.enums.strings import Loggers
from shared.types import Observation, IntegratedResult
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
        return {
            'neural_data': data_service.get_neural_sequence(),
            'symbolic_data': data_service.get_latest_data()
        }

    def process(self, observation: Observation) -> Optional[IntegratedResult]:
        neural_result = self.neural_module.predict(observation.get("neural_data", {}))
        # symbolic_result = self.symbolic_module.infer(observation.get("symbolic_data", {}))
        #
        # integrated_result = self.integration_strategy.integrate(
        #     neural_result,
        #     symbolic_result,
        #     self.confidence_weights
        # )

        return neural_result
