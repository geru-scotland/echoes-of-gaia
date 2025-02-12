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

class ComponentType(EnumBase):
    GROWTH = "growth"
    MOVEMENT = "movement"
    STATE = "state"
    CLIMATE = "climate"

class EntityState(EnumBase):
    HEALTHY = "healthy"
    WEAK = "weak"
    DYING = "dying"

class BiomeEvent(EnumBase):
    CLIMATE_CHANGE = "climate_change"
    SEASON_CHANGE = "season_change"
    DISASTER = "disaster"

class EntityType(EnumBase):
    FLORA = "flora"
    FAUNA = "fauna"
    HUMAN = "human"