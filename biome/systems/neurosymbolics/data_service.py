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
import os
from typing import Dict, List, Any, Optional
import time
from collections import deque
import numpy as np
from logging import Logger

import yaml

from biome.systems.events.event_bus import BiomeEventBus
from shared.enums.events import BiomeEvent
from utils.loggers import LoggerManager
from shared.enums.strings import Loggers
from utils.paths import DEEP_LEARNING_CONFIG_DIR


class NeurosymbolicDataService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = NeurosymbolicDataService()
        return cls._instance

    def __init__(self, history_length: int = 100):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._history_length = history_length
        self._neural_data_history: deque = deque(maxlen=history_length)
        self._species_data_history: deque = deque(maxlen=history_length)
        self._last_update_time = 0
        self._save_to_files = True
        self._config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        config_file: str = os.path.join(DEEP_LEARNING_CONFIG_DIR, 'neural_config.yaml')

        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        return config

    def update_data(self, neural_data: Dict[str, Any], species_data: Dict[str, Dict[str, Any]],
                    save_to_files: bool = True) -> None:
        self._neural_data_history.append(neural_data)
        self._species_data_history.append(species_data)
        self._last_update_time = time.time()
        self._save_to_files = save_to_files

        seq_length: int = self._config.get("data", {}).get("sequence_length")

        self._logger.info(f"Neural data history length: {len(self._neural_data_history)} / {seq_length} (seq length)")
        if 0 < seq_length <= len(self._neural_data_history):
            BiomeEventBus.trigger(BiomeEvent.NEUROSYMBOLIC_SERVICE_READY)

        self._logger.debug(f"Neurosymbolic data updated. History size: {len(self._neural_data_history)}")

    def get_neural_sequence(self, sequence_length: int = None) -> np.ndarray:
        seq_len = sequence_length or self._config["data"]["sequence_length"]
        seq_len = min(seq_len, len(self._neural_data_history))

        if seq_len == 0:
            return None

        # Hardcodeo por ahora. Features relevantes para la LSTM
        # Note: Ya he preparado config en módulo research, preparar bien soporte para inferencia
        features = self._config.get("data", {}).get("features")

        # Preparo la matriz de features: (seq_len, num_features)
        sequence = []
        for i in range(-seq_len, 0):
            data_point = self._neural_data_history[i]
            features_vector = [data_point.get(feature, 0) for feature in features]
            sequence.append(features_vector)
        return np.array(sequence)

    def get_graph_data(self) -> Dict[str, Any]:
        biome_latest_state: Dict[str, Any] = self._neural_data_history[-1]
        biome_features: List[str] = self._config.get("data", {}).get("graph_data", {}).get("biome_data", [])

        graph_data: Dict[str, Any] = {
            'biome_data': {
                feature: biome_latest_state[feature]
                for feature in self._config.get("data", {}).get("features", [])
                if feature in biome_features
            },
            'species_data': self._species_data_history[-1]
        }

        return graph_data

    def get_latest_data(self) -> Dict[str, Any]:
        if not self._species_data_history:
            return {}

        return {
            'neural_data': self._neural_data_history[-1] if self._neural_data_history else {},
            'species_data': self._species_data_history[-1]
        }

    def get_data_age(self) -> float:
        return time.time() - self._last_update_time

    def should_save_to_files(self) -> bool:
        return self._save_to_files

    def sequence_length(self) -> int:
        return self._config.get("hyperparameters", {}).get("sequence_length", 10)

    def clear_sequence_history(self) -> None:
        self._neural_data_history.clear()
        self._species_data_history.clear()
