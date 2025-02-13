from shared.base import EnumBaseStr, EnumBaseInt

# Maps
class TerrainType(EnumBaseStr):
    OCEAN_DEEP = 0
    OCEAN_MID = 1
    OCEAN_SHALLOW = 2
    BEACH = 3
    GRASS = 4
    MOUNTAIN = 5
    SNOW = 6

class ComponentType(EnumBaseStr):
    GROWTH = "growth"
    MOVEMENT = "movement"
    STATE = "state"
    CLIMATE = "climate"

class EntityState(EnumBaseStr):
    HEALTHY = "healthy"
    WEAK = "weak"
    DYING = "dying"

class BiomeEvent(EnumBaseStr):
    CLIMATE_CHANGE = "climate_change"
    SEASON_CHANGE = "season_change"
    DISASTER = "disaster"

class EntityType(EnumBaseStr):
    FLORA = "flora"
    FAUNA = "fauna"
    HUMAN = "human"

class FloraType(EnumBaseStr):
    pass

class FaunaType(EnumBaseStr):
    pass

class Timers:
    class Agents(EnumBaseInt):
        EVOLVE = 500

    class Entity(EnumBaseInt):
        GROWTH = 25
        ENERGY = 3

    class Biome(EnumBaseInt):
        CLIMATE = 50
        TEMPERATURE = 10

    class Simulation(EnumBaseInt):
        MONTH = 30