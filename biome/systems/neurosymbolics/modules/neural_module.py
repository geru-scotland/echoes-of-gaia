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
from typing import Protocol, Dict, Any

import numpy as np

from research.training.deep_learning.model_manager import NeuralModelManager
from shared.enums.enums import NeuralMode
from shared.enums.strings import Loggers
from shared.types import Observation, PredictionResult
from utils.loggers import LoggerManager


class NeuralModuleInterface(Protocol):
    def predict(self, observation: Observation) -> PredictionResult:
        ...

    def update(self, feedback: Dict[str, Any]) -> None:
        ...


class NeuralModule:
    def __init__(self, model_path: str = None):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info(f"Initalising Neural module...")
        self.model_manager = NeuralModelManager(mode=NeuralMode.INFERENCE)
        self._logger.info(f"Neural module ready")

    def predict(self, neural_data: Observation) -> PredictionResult:
        sequence: np.ndarray = neural_data
        if sequence is None or sequence.shape[0] < 5:
            return {"error": "Insufficient data for prediction"}

        prediction = self.model_manager.predict(sequence)

        return self._format_prediction(prediction)

    def _format_prediction(self, raw_prediction) -> PredictionResult:
        pass
