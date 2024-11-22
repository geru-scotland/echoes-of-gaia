import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import binary_dilation, gaussian_filter
from matplotlib.patches import Patch
import csv

grid_size = 75
scale = 2.0
octaves = 2
persistence = 1
lacunarity = 2.0

terrains = [
    {'name': 'agua', 'threshold': 0.4, 'target_percentage': 0.2, 'color': (0, 0, 1)},
    {'name': 'tierra', 'threshold': 0.7, 'target_percentage': 0.6, 'color': (0, 1, 0)},
    {'name': 'tecnología', 'threshold': 0.8, 'target_percentage': 0.2, 'color': (0.5, 0.5, 0.5)},
]


def generate_map(grid_size, scale, octaves, persistence, lacunarity):
    noise_map = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            nx = x / grid_size * scale
            ny = y / grid_size * scale
            noise_map[x][y] = pnoise2(nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                      repeatx=grid_size, repeaty=grid_size, base=43)
    return noise_map


def normalize_map(noise_map):
    min_val = np.min(noise_map)
    max_val = np.max(noise_map)
    return (noise_map - min_val) / (max_val - min_val)


def map_terrain_with_control(noise_map, terrains):
    terrain_map = np.zeros_like(noise_map, dtype=int)
    for i, terrain in enumerate(terrains):
        if i == 0:
            terrain_map[noise_map < terrain['threshold']] = i
        else:
            prev_threshold = terrains[i - 1]['threshold']
            terrain_map[(noise_map >= prev_threshold) & (noise_map < terrain['threshold'])] = i
    return terrain_map


def plot_map(terrain_map, terrains):
    color_map = np.zeros((terrain_map.shape[0], terrain_map.shape[1], 3))
    for i, terrain in enumerate(terrains):
        color_map[terrain_map == i] = terrain['color']

    plt.figure(figsize=(8, 8))
    plt.imshow(color_map, origin='upper')
    plt.title("Mapa Procedural Controlado")



    legend_elements = [
        Patch(facecolor=terrain['color'], label=terrain['name'].capitalize()) for terrain in terrains
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.axis('off')
    plt.show()


def save_map_to_csv(terrain_map, filename="controlled_terrain_map.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in terrain_map:
            writer.writerow(row)
    print(f"Mapa guardado en {filename}")


noise_map = generate_map(grid_size, scale, octaves, persistence, lacunarity)
normalized_map = normalize_map(noise_map)
terrain_map = map_terrain_with_control(normalized_map, terrains)

plot_map(terrain_map, terrains)
save_map_to_csv(terrain_map)
