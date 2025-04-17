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
                 stride: Optional[int] = None):
        self._sequence_length = sequence_length
        self._stride = stride if stride is not None else 1
        self._features = features
        self._targets = targets
        self._transform = transform
        self._simulation_boundaries = simulation_boundaries or []

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

        if not self._simulation_boundaries:
            sim_ranges = [(0, len(self._features_data))]
        else:
            sim_ranges = []
            for i, boundary in enumerate(self._simulation_boundaries):
                start_idx = self._simulation_boundaries[i - 1] if i > 0 else 0
                end_idx = boundary
                sim_ranges.append((start_idx, end_idx))
            sim_ranges.append((self._simulation_boundaries[-1], len(self._features_data)))

        for start_idx, end_idx in sim_ranges:
            segment_features = self._features_data[start_idx:end_idx]
            segment_targets = self._targets_data_raw[start_idx:end_idx]
            segment_length = len(segment_features)

            max_start = segment_length - self._sequence_length
            if max_start <= 0:
                continue

            for window_start in range(0, max_start + 1, self._stride):
                window_end = window_start + self._sequence_length
                seq_features = segment_features[window_start:window_end]

                target_index = window_end
                if target_index < segment_length:
                    target_values = segment_targets[target_index]
                    self._sequences.append(seq_features)
                    self._targets_data.append(target_values)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        sequence = self._sequences[idx]
        target = self._targets_data[idx]
        if self._transform:
            sequence = self._transform(sequence)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def _normalize_data(self):
        fmins = np.min(self._features_data, axis=0)
        fmaxs = np.max(self._features_data, axis=0)
        frange = fmaxs - fmins
        frange[frange == 0] = 1.0
        self._feature_stats = {'mins': fmins, 'maxs': fmaxs}
        self._features_data = (self._features_data - fmins) / frange

        tmins = np.min(self._targets_data_raw, axis=0)
        tmaxs = np.max(self._targets_data_raw, axis=0)
        trange = tmaxs - tmins
        trange[trange == 0] = 1.0
        self._target_stats = {'mins': tmins, 'maxs': tmaxs}
        self._targets_data_raw = (self._targets_data_raw - tmins) / trange

    def get_normalization_stats(self) -> Dict[str, Any]:
        return {'features': self._feature_stats, 'targets': self._target_stats}
