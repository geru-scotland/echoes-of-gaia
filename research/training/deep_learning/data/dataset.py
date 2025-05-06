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
                 target_horizon: int = 1,
                 normalize: bool = True,
                 normalization_stats: Optional[Dict[str, Any]] = None,
                 normalization_method: str = 'minmax',
                 smoothing_window: int = 35):

        self._sequence_length = sequence_length
        self._stride = sequence_length // 2
        self._features = features
        self._targets = targets
        self._transform = transform
        self._simulation_boundaries = simulation_boundaries or []
        self._target_horizon = target_horizon
        self._sequences = []
        self._targets_data = []
        self._feature_stats = {}
        self._target_stats = {}
        self._normalization_method = normalization_method
        self._apply_smoothing = False
        self._smoothing_window = smoothing_window if smoothing_window else 15

        if isinstance(data, list) and isinstance(data[0], dict):
            self._features_data, self._targets_data_raw = self._preprocess_dict_data(data)
        else:
            self._features_data = np.array(data, dtype=float)
            self._targets_data_raw = np.array(data, dtype=float)

        if normalize:
            if normalization_stats:
                self._feature_stats = normalization_stats['features']
                self._target_stats = normalization_stats['targets']
                self._normalize_with_existing_stats(self._normalization_method)
            else:
                self._normalize_data(self._normalization_method)
        else:
            self._targets_data_norm = self._targets_data_raw.copy()

        self._create_sequences()

    def _normalize_with_existing_stats(self, method='minmax'):
        if method == 'minmax':
            feature_mins = self._feature_stats['mins']
            feature_maxs = self._feature_stats['maxs']
            feature_range = feature_maxs - feature_mins
            feature_range[feature_range == 0] = 1.0
            self._features_data = (self._features_data - feature_mins) / feature_range

            target_mins = self._target_stats['mins']
            target_maxs = self._target_stats['maxs']
            target_range = target_maxs - target_mins
            target_range[target_range == 0] = 1.0
            self._targets_data_norm = (self._targets_data_raw - target_mins) / target_range

        elif method == 'zscore':
            feature_means = self._feature_stats['means']
            feature_stds = self._feature_stats['stds']
            self._features_data = (self._features_data - feature_means) / feature_stds

            target_means = self._target_stats['means']
            target_stds = self._target_stats['stds']
            self._targets_data_norm = (self._targets_data_raw - target_means) / target_stds

        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'standard'.")

    def _preprocess_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        features_list, targets_list = [], []
        for record in data:
            features_list.append([float(record.get(f, 0.0)) for f in self._features])
            targets_list.append([float(record.get(t, 0.0)) for t in self._targets])

        features_array = np.array(features_list)
        targets_array = np.array(targets_list)

        if self._apply_smoothing:
            features_array = self._apply_ema(features_array)

        return features_array, targets_array

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
                for i in range(0, len(sim_features) - self._sequence_length - self._target_horizon, self._stride):
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

    def _normalize_data(self, method='minmax'):
        feature_mins = np.min(self._features_data, axis=0)
        feature_maxs = np.max(self._features_data, axis=0)
        feature_range = feature_maxs - feature_mins
        feature_range[feature_range == 0] = 1.0

        feature_means = np.mean(self._features_data, axis=0)
        feature_stds = np.std(self._features_data, axis=0)
        feature_stds[feature_stds == 0] = 1.0

        self._feature_stats = {
            'mins': feature_mins,
            'maxs': feature_maxs,
            'means': feature_means,
            'stds': feature_stds
        }

        target_mins = np.min(self._targets_data_raw, axis=0)
        target_maxs = np.max(self._targets_data_raw, axis=0)
        target_range = target_maxs - target_mins
        target_range[target_range == 0] = 1.0

        target_means = np.mean(self._targets_data_raw, axis=0)
        target_stds = np.std(self._targets_data_raw, axis=0)
        target_stds[target_stds == 0] = 1.0

        self._target_stats = {
            'mins': target_mins,
            'maxs': target_maxs,
            'means': target_means,
            'stds': target_stds
        }

        if method == 'minmax':
            self._features_data = (self._features_data - feature_mins) / feature_range
            self._targets_data_norm = (self._targets_data_raw - target_mins) / target_range
        elif method == 'zscore':
            self._features_data = (self._features_data - feature_means) / feature_stds
            self._targets_data_norm = (self._targets_data_raw - target_means) / target_stds
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'minmax' or 'standard'.")

    def get_normalization_stats(self) -> Dict[str, Any]:
        return {'features': self._feature_stats, 'targets': self._target_stats}

    def _apply_moving_average(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        smoothed_data = np.zeros_like(data)

        sim_ranges = []
        if not self._simulation_boundaries:
            sim_ranges = [(0, len(data))]
        else:
            for i in range(len(self._simulation_boundaries)):
                start = self._simulation_boundaries[i - 1] if i > 0 else 0
                end = self._simulation_boundaries[i]
                sim_ranges.append((start, end))

        for start, end in sim_ranges:
            for i in range(start, end):
                window_start = max(start, i - window_size // 2)
                window_end = min(end, i + window_size // 2 + 1)

                smoothed_data[i] = np.mean(data[window_start:window_end], axis=0)

        return smoothed_data

    def _apply_ema(self, data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        smoothed_data = np.zeros_like(data)

        sim_ranges = []
        if not self._simulation_boundaries:
            sim_ranges = [(0, len(data))]
        else:
            for i in range(len(self._simulation_boundaries)):
                start = self._simulation_boundaries[i - 1] if i > 0 else 0
                end = self._simulation_boundaries[i]
                sim_ranges.append((start, end))

        for start, end in sim_ranges:
            smoothed_data[start] = data[start]

            for i in range(start + 1, end):
                smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]

        return smoothed_data
