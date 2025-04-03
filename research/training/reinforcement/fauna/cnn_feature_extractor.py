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
            nn.Conv2d(embedding_dim + 2, 32, kernel_size=3, stride=1, padding=1),
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

        self.print_terrain_embeddings()

    def forward(self, observations) -> th.Tensor:

        # Para aclararme (Batch, width, height)
        terrain_indices = observations["terrain_map"].long()
        valid_map = observations["validity_map"]
        visited_map = observations["visited_map"]

        max_index = len(list(TerrainType)) - 1
        terrain_indices = torch.clamp(terrain_indices, 0, max_index)

        terrain_embedded = self.terrain_embedding(terrain_indices)
        # Al pasar por capa de embeddings: (Batch, width, height, num_embeddings)
        # Reordeno dimensiones para que los num_embedding sean los canales y estén en dim 1
        # (Batch, channels/num_embedd, width, hegiht)
        terrain_embedded = terrain_embedded.permute(0, 3, 1, 2)

        # Agrego dimensión de canal: (Batch, 1, width, height)
        validity_channel = valid_map.unsqueeze(1)
        visited_channel = visited_map.unsqueeze(1)
        # +1 canal (lámina en el stack), le agrego a num_embedd + 1 con el validity channel + 1 visited
        combined_maps = torch.cat([terrain_embedded, validity_channel, visited_channel], dim=1)

        cnn_features = self.cnn(combined_maps)

        return self.linear(cnn_features)

        # print(f"\n=== TERRAIN MAP STATS ===")
        # print(f"Shape: {terrain_indices.shape}")
        # print(f"Min: {terrain_indices.min().item()}, Max: {terrain_indices.max().item()}")
        # print(f"Unique values: {torch.unique(terrain_indices).tolist()}")
        #
        # print(f"\n=== VALIDITY MAP STATS ===")
        # print(f"Shape: {valid_map.shape}")
        # print(
        #     f"Valid positions: {valid_map.sum().item()}/{valid_map.numel()} ({valid_map.sum().item() / valid_map.numel() * 100:.2f}%)")
        #
        # if terrain_indices.shape[0] > 0:
        #     # Crear mapas visuales para el primer elemento del batch
        #     print("\n=== FIRST TERRAIN MAP IN BATCH ===")
        #     terrain_map = terrain_indices[0].cpu().numpy()
        #
        #     terrain_names = {
        #         0: "WD",  # WATER_DEEP
        #         1: "WM",  # WATER_MID
        #         2: "WS",  # WATER_SHALLOW
        #         3: "SH",  # SHORE
        #         4: "GR",  # GRASS
        #         5: "MT",  # MOUNTAIN
        #         6: "SN",  # SNOW
        #         7: "SD",  # SAND
        #         8: "UN",  # UNKNOWN
        #     }
        #
        #     for row in terrain_map:
        #         row_str = " ".join([f"{terrain_names.get(int(val), '??'):2}" for val in row])
        #         print(row_str)
        #
        #     print("\n=== FIRST VALIDITY MAP IN BATCH ===")
        #     validity_map = valid_map[0].cpu().numpy()
        #
        #     for row in validity_map:
        #         row_str = " ".join(["✓" if val > 0.5 else "✗" for val in row])
        #         print(row_str)
        #
        #     center_y, center_x = terrain_map.shape[0] // 2, terrain_map.shape[1] // 2
        #     print(f"\nPosición del agente (centro): y={center_y}, x={center_x}")
        #     print(f"Terreno en posición del agente: {terrain_names.get(int(terrain_map[center_y, center_x]), '??')}")
        #     print(f"¿Es válida la posición del agente?: {'Sí' if validity_map[center_y, center_x] > 0.5 else 'No'}")
        #
        #     top_edge = validity_map[0]
        #     bottom_edge = validity_map[-1]
        #     left_edge = validity_map[:, 0]
        #     right_edge = validity_map[:, -1]
        #
        #     print("\n=== ANÁLISIS DE BORDES ===")
        #     print(f"Borde superior: {sum(top_edge)}/{len(top_edge)} posiciones válidas")
        #     print(f"Borde inferior: {sum(bottom_edge)}/{len(bottom_edge)} posiciones válidas")
        #     print(f"Borde izquierdo: {sum(left_edge)}/{len(left_edge)} posiciones válidas")
        #     print(f"Borde derecho: {sum(right_edge)}/{len(right_edge)} posiciones válidas")
        #
        # self.print_terrain_index_distribution(terrain_indices)

    def save_terrain_embeddings(self, path: str = "./"):
        import os
        import numpy as np

        os.makedirs(path, exist_ok=True)

        terrain_names = {
            0: "WATER_DEEP",
            1: "WATER_MID",
            2: "WATER_SHALLOW",
            3: "SHORE",
            4: "GRASS",
            5: "MOUNTAIN",
            6: "SNOW",
            7: "SAND",
            # 8: "UNKNWON",  # opcional
        }

        embedding_weights = self.terrain_embedding.weight.detach().cpu().numpy()

        metadata_path = os.path.join(path, "metadata.tsv")
        vectors_path = os.path.join(path, "vectors.tsv")

        with open(metadata_path, "w") as f:
            for idx in terrain_names:
                f.write(f"{terrain_names[idx]}\n")

        with open(vectors_path, "w") as f:
            for idx in terrain_names:
                vector = "\t".join(f"{x:.8f}" for x in embedding_weights[idx])
                f.write(f"{vector}\n")

        print(f"Embeddings stored {metadata_path} and {vectors_path}")

    def print_terrain_embeddings(self):
        import pandas as pd

        embedding_weights = self.terrain_embedding.weight.detach().cpu().numpy()

        terrain_names = {
            0: "WATER_DEEP",
            1: "WATER_MID",
            2: "WATER_SHALLOW",
            3: "SHORE",
            4: "GRASS",
            5: "MOUNTAIN",
            6: "SNOW",
            7: "SAND",
            8: "UNKNWON",
        }

        rows = []
        for idx, name in terrain_names.items():
            vector = embedding_weights[idx]
            rows.append({"Index": idx, "Name": name, "Embedding": vector})

        df = pd.DataFrame(rows)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None)
        print(df)

    def print_terrain_index_distribution(self, terrain_indices: torch.Tensor):
        import pandas as pd
        import numpy as np

        terrain_names = {
            0: "WATER_DEEP",
            1: "WATER_MID",
            2: "WATER_SHALLOW",
            3: "SHORE",
            4: "GRASS",
            5: "MOUNTAIN",
            6: "SNOW",
            7: "SAND",
            8: "UNKNWON",
        }

        flat_indices = terrain_indices.detach().cpu().numpy().flatten()
        unique, counts = np.unique(flat_indices, return_counts=True)

        data = [{"Index": int(idx), "Name": terrain_names.get(int(idx), "UNKNOWN"), "Count": int(count)}
                for idx, count in zip(unique, counts)]

        df = pd.DataFrame(data).sort_values("Count", ascending=False).reset_index(drop=True)
        print("\n[Terrain Frequency in Current Batch]")
        print(df)


def create_custom_cnn_policy():
    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    return policy_kwargs
