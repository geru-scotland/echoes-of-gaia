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
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SimulationDataset(Dataset):
    def __init__(self,
                 data: Union[List[Dict], np.ndarray],
                 sequence_length: int,
                 features: List[str],
                 targets: List[str],
                 simulation_boundaries: Optional[List[int]] = None,
                 transform=None,
                 stride: Optional[int] = None,
                 target_horizon: int = 1):
        self._sequence_length = sequence_length
        self._stride = stride if stride is not None else 1
        self._features = features
        self._targets = targets
        self._transform = transform
        self._simulation_boundaries = simulation_boundaries or []
        self._target_horizon = target_horizon
        self._sequences = []
        self._targets_data = []
        self._feature_stats = {}
        self._target_stats = {}

        if isinstance(data, list) and isinstance(data[0], dict):
            self._features_data, self._targets_data_raw = self._preprocess_dict_data(data)
        else:
            self._features_data = np.array(data, dtype=float)
            self._targets_data_raw = np.array(data, dtype=float)

        self._normalize_data()
        self._create_sequences()

    def _preprocess_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        features_list, targets_list = [], []
        for record in data:
            features_list.append([float(record.get(f, 0.0)) for f in self._features])
            targets_list.append([float(record.get(t, 0.0)) for t in self._targets])
        return np.array(features_list), np.array(targets_list)

    def _create_sequences(self):
        self._sequences.clear()
        self._targets_data.clear()

        if not self._simulation_boundaries:
            sim_ranges = [(0, len(self._features_data))]
        else:
            sim_ranges = []
            for i in range(len(self._simulation_boundaries)):
                start = self._simulation_boundaries[i - 1] if i > 0 else 0
                end = self._simulation_boundaries[i]
                sim_ranges.append((start, end))

        for start, end in sim_ranges:
            sim_features = self._features_data[start:end]
            sim_targets = self._targets_data_norm[start:end]

            if len(sim_features) > self._sequence_length + self._target_horizon:
                for i in range(len(sim_features) - self._sequence_length - self._target_horizon):
                    seq_features = sim_features[i:i + self._sequence_length]

                    # Múltiples targets consecutivos, ddetermino con horizon
                    target_values = np.zeros((self._target_horizon, len(self._targets)))
                    for step in range(self._target_horizon):
                        target_idx = i + self._sequence_length + step
                        target_values[step] = sim_targets[target_idx]

                    self._sequences.append(seq_features)
                    self._targets_data.append(target_values)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self._sequences[idx]
        target = self._targets_data[idx]

        if self._transform:
            sequence = self._transform(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def _normalize_data(self):
        feature_mins = np.min(self._features_data, axis=0)
        feature_maxs = np.max(self._features_data, axis=0)
        feature_range = feature_maxs - feature_mins
        feature_range[feature_range == 0] = 1.0
        self._feature_stats = {'mins': feature_mins, 'maxs': feature_maxs}
        self._features_data = (self._features_data - feature_mins) / feature_range

        target_mins = np.min(self._targets_data_raw, axis=0)
        target_maxs = np.max(self._targets_data_raw, axis=0)
        target_range = target_maxs - target_mins
        target_range[target_range == 0] = 1.0
        self._target_stats = {'mins': target_mins, 'maxs': target_maxs}

        self._targets_data_norm = (self._targets_data_raw - target_mins) / target_range

    def get_normalization_stats(self) -> Dict[str, Any]:
        return {'features': self._feature_stats, 'targets': self._target_stats}
