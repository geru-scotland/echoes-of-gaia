from shared.enums import TerrainType
# Biome
MAP_DEFAULT_SIZE = (50, 50)
# TODO: Llevar al config
BIOME_TYPE_WEIGHTS = {
    "tropical": [10, 10, 19, 5, 55, 30, 5]
}
TERRAIN_TYPES = [
    TerrainType.OCEAN_DEEP, TerrainType.OCEAN_MID, TerrainType.OCEAN_SHALLOW,
    TerrainType.BEACH, TerrainType.GRASS, TerrainType.MOUNTAIN, TerrainType.SNOW
]