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
import gym
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from shared.enums.enums import TerrainType


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, embedding_dim: int = 8):
        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim)

        local_map_space = observation_space.spaces["terrain_map"]
        num_terrain_types = len(list(TerrainType))

        self.terrain_embedding = nn.Embedding(
            num_embeddings=num_terrain_types,
            embedding_dim=embedding_dim,
            padding_idx=TerrainType.UNKNWON.value
        )

        h, w = local_map_space.shape

        self.cnn = nn.Sequential(
            nn.Conv2d(embedding_dim + 1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out_size = 64 * h * w

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        terrain_indices = observations["terrain_map"].long()
        max_index = len(list(TerrainType)) - 1
        terrain_indices = torch.clamp(terrain_indices, 0, max_index)

        valid_mask = observations["validity_map"]

        terrain_embedded = self.terrain_embedding(terrain_indices)
        terrain_embedded = terrain_embedded.permute(0, 3, 1, 2)

        validity_channel = valid_mask.unsqueeze(1)
        combined_maps = torch.cat([terrain_embedded, validity_channel], dim=1)

        cnn_features = self.cnn(combined_maps)

        return self.linear(cnn_features)


def create_custom_cnn_policy():
    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    return policy_kwargs
