from shared.enums import TerrainType
# Biome
MAP_DEFAULT_SIZE = (50, 50)
BIOME_TYPE_WEIGHTS = {
    "tropical": [40, 10, 10, 12, 55, 30, 0]
}
TERRAIN_TYPES = [
    TerrainType.OCEAN_DEEP, TerrainType.OCEAN_MID, TerrainType.OCEAN_SHALLOW,
    TerrainType.BEACH, TerrainType.GRASS, TerrainType.MOUNTAIN, TerrainType.SNOW
]