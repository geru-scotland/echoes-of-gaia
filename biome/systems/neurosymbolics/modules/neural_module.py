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
        self._logger.debug(f"Showing latest 5 observations from sequence: {sequence[5:]}")

        prediction_numpy = self.model_manager.predict(sequence)
        self._logger.debug(
            f"Prediction (type: {type(prediction_numpy)}, shape: {prediction_numpy.shape}): {prediction_numpy}")

        prediction = np.squeeze(prediction_numpy, axis=0)
        prediction_avg = list(np.mean(prediction, axis=0))

        self._logger.debug(
            f"Prediction (type: {type(prediction)}, shape: {prediction.shape}): {prediction}")
        self._logger.debug(
            f"Prediction averages): {prediction_avg}")

        return self._format_prediction(prediction_avg)

    def _format_prediction(self, raw_prediction) -> PredictionResult:
        return {
            'prey_population': raw_prediction[0],
            'predator_population': raw_prediction[1],
            'flora_count': raw_prediction[2]
        }
