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

from typing import Dict, List, Any, Optional
import time
from collections import deque
import numpy as np
from logging import Logger

from utils.loggers import LoggerManager
from shared.enums.strings import Loggers


class NeurosymbolicDataService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NeurosymbolicDataService()
        return cls._instance

    def __init__(self, history_length: int = 10):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._history_length = history_length
        self._lstm_data_history: deque = deque(maxlen=history_length)
        self._species_data_history: deque = deque(maxlen=history_length)
        self._last_update_time = 0
        self._save_to_files = True

    def update_data(self, lstm_data: Dict[str, Any], species_data: Dict[str, Dict[str, Any]],
                    save_to_files: bool = True) -> None:
        self._lstm_data_history.append(lstm_data)
        self._species_data_history.append(species_data)
        self._last_update_time = time.time()
        self._save_to_files = save_to_files

        self._logger.debug(f"Neurosymbolic data updated. History size: {len(self._lstm_data_history)}")

    def get_lstm_sequence(self, sequence_length: int = None) -> np.ndarray:
        seq_len = sequence_length or len(self._lstm_data_history)
        seq_len = min(seq_len, len(self._lstm_data_history))

        if seq_len == 0:
            return None

        # Hardcodeo por ahora. Features relevantes para la LSTM
        features = [
            'prey_population', 'predator_population', 'predator_prey_ratio',
            'avg_stress', 'avg_energy', 'ecosystem_score', 'biodiversity_index',
            'herbivore_pop', 'carnivore_pop', 'primary_producer_pop',
            'temperature', 'humidity', 'precipitation'
        ]

        # Preparo la matriz de features: (seq_len, num_features)
        sequence = []
        for i in range(-seq_len, 0):
            data_point = self._lstm_data_history[i]
            features_vector = [data_point.get(feature, 0) for feature in features]
            sequence.append(features_vector)

        return np.array(sequence)

    def get_latest_graph_data(self) -> Dict[str, Any]:
        if not self._species_data_history:
            return {}

        return {
            'lstm_data': self._lstm_data_history[-1] if self._lstm_data_history else {},
            'species_data': self._species_data_history[-1]
        }

    def get_data_age(self) -> float:
        return time.time() - self._last_update_time

    def should_save_to_files(self) -> bool:
        return self._save_to_files
