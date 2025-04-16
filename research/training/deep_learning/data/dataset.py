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
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class SimulationDataset(Dataset):
    def __init__(self,
                 data: Union[List[Dict], np.ndarray],
                 sequence_length: int,
                 features: List[str],
                 targets: List[str],
                 simulation_boundaries: Optional[List[int]] = None,
                 transform=None):
        self._sequence_length = sequence_length
        self._features = features
        self._targets = targets
        self._transform = transform
        self._simulation_boundaries = simulation_boundaries
        self._sequences = []
        self._targets_data = []
        self._feature_stats = {}
        self._target_stats = {}

        self.augment_climate = True
        self.climate_features_idx = [5, 6, 7, 8, 9]
        self.noise_std = 0.05

        if isinstance(data, list) and isinstance(data[0], dict):
            self._features_data, self._targets_data_raw = self._preprocess_dict_data(data)
        else:
            self._features_data = data
            self._targets_data_raw = data

        self._normalize_data()
        self._create_sequences()

    def _preprocess_dict_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        features_data = []
        targets_data = []

        for item in data:
            features_values = [float(item.get(feature, 0.0)) for feature in self._features]
            features_data.append(features_values)

            target_values = [float(item.get(target, 0.0)) for target in self._targets]
            targets_data.append(target_values)

        return np.array(features_data), np.array(targets_data)

    def _create_sequences(self):
        if not self._simulation_boundaries:
            sim_ranges = [(0, len(self._features_data))]
        else:
            sim_ranges = []
            for i in range(len(self._simulation_boundaries)):
                start = self._simulation_boundaries[i - 1] if i > 0 else 0
                end = self._simulation_boundaries[i] if i < len(self._simulation_boundaries) else len(
                    self._features_data)
                sim_ranges.append((start, end))

        for start, end in sim_ranges:
            sim_features = self._features_data[start:end]
            sim_targets = self._targets_data_raw[start:end]

            if len(sim_features) > self._sequence_length:
                for i in range(len(sim_features) - self._sequence_length - 1):
                    seq_features = sim_features[i:i + self._sequence_length]

                    target_idx = i + self._sequence_length
                    if target_idx < len(sim_targets):
                        target_values = sim_targets[target_idx]

                        self._sequences.append(seq_features)
                        self._targets_data.append(target_values)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self._sequences[idx]
        target = self._targets_data[idx]

        if self._transform:
            sequence = self._transform(sequence)

        # if self.augment_climate and self._transform is None:
        #     sequence = self._add_climate_noise(sequence)

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def _normalize_data(self):
        feature_means = np.mean(self._features_data, axis=0)
        feature_stds = np.std(self._features_data, axis=0)

        feature_stds[feature_stds == 0] = 1.0

        self._feature_stats = {
            'means': feature_means,
            'stds': feature_stds
        }

        self._features_data = (self._features_data - feature_means) / feature_stds

        target_means = np.mean(self._targets_data_raw, axis=0)
        target_stds = np.std(self._targets_data_raw, axis=0)

        target_stds[target_stds == 0] = 1.0

        self._target_stats = {
            'means': target_means,
            'stds': target_stds
        }

        self._targets_data_raw = (self._targets_data_raw - target_means) / target_stds

    def _add_climate_noise(self, sequence):
        noisy_seq = sequence.copy()
        for t in range(sequence.shape[0]):
            noise = np.random.normal(0, self.noise_std, size=len(self.climate_features_idx))
            noisy_seq[t, self.climate_features_idx] += noise
        return noisy_seq

    def get_normalization_stats(self):
        return {
            'features': self._feature_stats,
            'targets': self._target_stats
        }
