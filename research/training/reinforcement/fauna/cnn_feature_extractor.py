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


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim)

        local_map_space = observation_space.spaces["local_map"]

        # Por ahora 2 canales, local map y exploration map
        # Pero seguramente haga 3, entity_map, terrain_map y exploration map
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out_size = 64 * local_map_space.shape[0] * local_map_space.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations) -> th.Tensor:
        local_map = observations["local_map"]
        exploration_map = observations["exploration_map"]

        combined_maps = torch.cat([
            local_map.unsqueeze(1),
            exploration_map.unsqueeze(1)
        ], dim=1)

        cnn_features = self.cnn(combined_maps)

        return self.linear(cnn_features)


def create_custom_cnn_policy():
    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    return policy_kwargs
