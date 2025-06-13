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

"""
CNN-based feature extractor for processing environmental observation spaces.

Implements a neural network architecture for processing multi-channel biome 
observations; combines terrain, entity, and physiological data through embeddings 
and convolutional layers. Includes visualization tools for terrain and biome 
representations - creates compact feature vectors for RL policy networks.
"""


import gym
import torch
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from shared.enums.enums import BiomeType, DietType, TerrainType


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, terrain_embedding_dim: int = 8,
                 biome_embedding_dim: int = 8, diet_embedding_dim: int = 8):
        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim)

        local_map_space = observation_space.spaces["terrain_map"]

        num_terrain_types = len(list(TerrainType))
        num_biome_types = len(list(BiomeType))
        num_diet_types = len(list(DietType))

        self.terrain_embedding = nn.Embedding(
            num_embeddings=num_terrain_types,
            embedding_dim=terrain_embedding_dim,
            padding_idx=TerrainType.UNKNWON.value
        )

        self.biome_embedding = nn.Embedding(
            num_embeddings=num_biome_types,
            embedding_dim=biome_embedding_dim
        )

        self.diet_embedding = nn.Embedding(
            num_embeddings=num_diet_types,
            embedding_dim=diet_embedding_dim
        )

        h, w = local_map_space.shape

        # TODO: Probar a poner una capa convolucional más. Jugar con los pasos y el padding
        # NOTA: Ojo, si agrego más pooling, que no se me pase el modificar h_pooled y w_pooled!
        self.cnn = nn.Sequential(
            nn.Conv2d(terrain_embedding_dim + 7, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )
        h_pooled = h // 2
        w_pooled = w // 2
        cnn_out_size = 64 * h_pooled * w_pooled

        biome_embedd_flat_size = biome_embedding_dim * num_biome_types
        diet_embedd_flat_size = diet_embedding_dim * num_diet_types  # Lo mismo, como no puedo aplicar convoluciones, al no ser espacial, concanteno

        # Se me está yendo un poco de madre ahora con tantos 1, pero lo pongo explícito
        # por claridad, cambiar quizá hacia el final esto.
        thirst_feature_size = 1
        energy_reserves_feature_size = 1
        vitality_feature_size = 1
        stress_level_feature_size = 1
        hunger_level_feature_size = 1
        somatic_integrity_feature_size = 1

        combined_features_size = (cnn_out_size + biome_embedd_flat_size + diet_embedd_flat_size +
                                  thirst_feature_size + energy_reserves_feature_size +
                                  vitality_feature_size + stress_level_feature_size + hunger_level_feature_size
                                  + somatic_integrity_feature_size)

        self.linear = nn.Sequential(
            nn.Linear(combined_features_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        # self.print_terrain_embeddings()
        # self.print_biome_embeddings()

    def forward(self, observations) -> th.Tensor:

        # Para aclararme (Batch, width, height)
        biome_index = observations["biome_type"].long()
        terrain_indices = observations["terrain_map"].long()
        diet_index = observations["diet_type"].long()
        valid_map = observations["validity_map"]
        visited_map = observations["visited_map"]
        flora_map = observations["flora_map"]
        prey_map = observations["prey_map"]
        predator_map = observations["predator_map"]
        water_map = observations["water_map"]
        food_map = observations["food_map"]
        thirst_level = observations["thirst_level"]
        energy_reserves = observations["energy_reserves"]
        vitality = observations["vitality"]
        stress_level = observations["stress_level"]
        hunger_level = observations["hunger_level"]
        somatic_integrity = observations["somatic_integrity"]

        # TODO: COMPROBAR BIEN LOS FAUNA Y FLORA MAP
        max_index = len(list(TerrainType)) - 1
        terrain_indices = torch.clamp(terrain_indices, 0, max_index)

        terrain_embedded = self.terrain_embedding(terrain_indices)
        # Al pasar por capa de embeddings: (Batch, width, height, num_embeddings)
        # Reordeno dimensiones para que los num_embedding sean los canales y estén en dim 1
        # (Batch, channels/num_embedd, width, hegiht)
        terrain_embedded = terrain_embedded.permute(0, 3, 1, 2)

        biome_embedded = self.biome_embedding(biome_index)
        diet_embedded = self.diet_embedding(diet_index)

        # Agrego dimensión de canal: (Batch, 1, width, height)
        validity_channel = valid_map.unsqueeze(1)
        visited_channel = visited_map.unsqueeze(1)
        water_channel = water_map.unsqueeze(1)
        food_channel = food_map.unsqueeze(1)

        flora_map = flora_map.unsqueeze(1)
        prey_map = prey_map.unsqueeze(1)
        predator_map = predator_map.unsqueeze(1)

        # Nota: a CNN, sólo mapas espaciales, disposición 2D.
        # +1 canal (lámina en el stack), le agrego a num_embedd + 1 con el validity channel + 1 visited
        combined_maps = torch.cat([terrain_embedded,
                                   validity_channel,
                                   visited_channel,
                                   flora_map,
                                   prey_map,
                                   predator_map,
                                   water_channel,
                                   food_channel
                                   ], dim=1)
        cnn_features = self.cnn(combined_maps)

        thirst_features = thirst_level.view(thirst_level.size(0), -1)
        energy_reserves_features = energy_reserves.view(energy_reserves.size(0), -1)
        biome_embedded = biome_embedded.view(biome_embedded.size(0), -1)
        diet_embedded = diet_embedded.view(diet_embedded.size(0), -1)
        vitality_features = vitality.view(vitality.size(0), -1)
        stress_level_features = stress_level.view(stress_level.size(0), -1)
        hunger_level_features = hunger_level.view(hunger_level.size(0), -1)
        somatic_integrity_features = somatic_integrity.view(somatic_integrity.size(0), -1)
        # print(f"[DEBUG] cnn_features shape: {cnn_features.shape}")
        # print(f"[DEBUG] biome_embedded shape: {biome_embedded.shape}")
        # print(f"[DEBUG] thirst_features shape: {thirst_features.shape}")

        combined_features = torch.cat([
            cnn_features, biome_embedded, thirst_features, energy_reserves_features,
            vitality_features, stress_level_features, hunger_level_features, diet_embedded,
            somatic_integrity_features
        ], dim=1)
        # print(f"[DEBUG] combined_features shape: {combined_features.shape}")

        # self.print_biome_embeddings()
        return self.linear(combined_features)

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
        from datetime import datetime

        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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

        metadata_path = os.path.join(path, f"terrain_metadata_{timestamp}.tsv")
        vectors_path = os.path.join(path, f"terrain_embedding_{timestamp}.tsv")

        with open(metadata_path, "w") as f:
            for idx in terrain_names:
                f.write(f"{terrain_names[idx]}\n")

        with open(vectors_path, "w") as f:
            for idx in terrain_names:
                vector = "\t".join(f"{x:.8f}" for x in embedding_weights[idx])
                f.write(f"{vector}\n")

        print(f"Terrain embeddings stored in {metadata_path} and {vectors_path}")

    def save_biome_embeddings(self, path: str = "./"):
        import os
        from datetime import datetime

        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        biome_names = {idx: biome.value for idx, biome in enumerate(BiomeType)}

        embedding_weights = self.biome_embedding.weight.detach().cpu().numpy()

        metadata_path = os.path.join(path, f"biome_metadata_{timestamp}.tsv")
        vectors_path = os.path.join(path, f"biome_embedding_{timestamp}.tsv")

        with open(metadata_path, "w") as f:
            for idx in range(len(biome_names)):
                f.write(f"{biome_names[idx]}\n")

        with open(vectors_path, "w") as f:
            for idx in range(len(biome_names)):
                vector = "\t".join(f"{x:.8f}" for x in embedding_weights[idx])
                f.write(f"{vector}\n")

        print(f"Biome embeddings stored in {metadata_path} and {vectors_path}")

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
        import numpy as np
        import pandas as pd

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

    def print_biome_embeddings(self):
        import pandas as pd

        embedding_weights = self.biome_embedding.weight.detach().cpu().numpy()

        biome_names = {
            0: "TROPICAL",
            1: "DESERT",
            2: "TAIGA",
            3: "SAVANNA",
            4: "TUNDRA",
        }

        rows = []
        for idx, name in biome_names.items():
            vector = embedding_weights[idx]
            rows.append({"Index": idx, "Name": name, "Embedding": vector})

        df = pd.DataFrame(rows)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', None)
        print("\n[Biome Embeddings]")
        print(df)


def create_custom_cnn_policy():
    policy_kwargs = dict(
        features_extractor_class=CNNFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256)
    )
    return policy_kwargs
