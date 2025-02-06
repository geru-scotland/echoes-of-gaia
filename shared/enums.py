from shared.base import EnumBase

class Strings(EnumBase):
     BIOME_BUILDER = "biome_builder"
     SIMULATION_BUILDER = "simulation_builder"

     BIOME_CONTEXT = "biome_ctx"
     SIMULATION_CONTEXT = "simulation_ctx"

# Maps
class TerrainType(EnumBase):
    OCEAN_DEEP = 0
    OCEAN_MID = 1
    OCEAN_SHALLOW = 2
    BEACH = 3
    GRASS = 4
    MOUNTAIN = 5
    SNOW = 6
