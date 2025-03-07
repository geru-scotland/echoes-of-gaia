"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from shared.base import EnumBaseStr, EnumBaseInt

class BiomeType(EnumBaseStr):
    TROPICAL = "tropical"
    DESERT = "desert"
    TAIGA = "taiga"
    SAVANNA = "savanna"
    TUNDRA = "tundra"

# Maps
class TerrainType(EnumBaseInt):
    OCEAN_DEEP = 0
    OCEAN_MID = 1
    OCEAN_SHALLOW = 2
    BEACH = 3
    GRASS = 4
    MOUNTAIN = 5
    SNOW = 6

class ComponentType(EnumBaseStr):
    TRANSFORM = "transform"
    GROWTH = "growth"
    NUTRITIONAL = "nutritional"
    MOVEMENT = "movement"
    STATE = "state"
    VITAL = "vital"
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

class EntityIndex(EnumBaseInt):
    FLORA = 1
    FAUNA = 2

class FloraSpecies(EnumBaseStr):
    OAK_TREE = "oak_tree"
    BRAMBLE = "bramble"
    MUSHROOM = "mushroom"

class FaunaSpecies(EnumBaseStr):
    DEER = "deer"
    BOAR = "boar"
    FOX = "fox"

class Timers:
    class Agents(EnumBaseInt):
        EVOLUTIONARY_CYCLE = 500
        CLIMATE_UPDATE = 2 # 2 ticks = 1 día

    class Entity(EnumBaseInt):
        GROWTH = 25
        NUTRITIONAL_VALUE_DECAY = 5
        ENERGY = 3
        AGING = 30
        HEALTH_DECAY = 5

    class Biome(EnumBaseInt):
        CLIMATE = 50
        TEMPERATURE = 10

    class Simulation(EnumBaseInt):
        MONTH = 30


class Habitats:
    class Relations(EnumBaseStr):
        IN = "IN"
        NEARBY = "NEARBY"

    class Type(EnumBaseStr):
        FOREST = "forest"
        THICKET = "thicket"
        FUNGAL = "fungal_zone"
        COASTAL = "coastal"
        ALPINE = "alpine"

class CapturePeriod(EnumBaseInt):
    MONTHLY = 60
    WEEKLY = 15
    DAILY = 2
    CUSTOM = -1

class CaptureFormat(EnumBaseStr):
    JSON = "json"
    JSONL = "jsonl"
    YAML = "yaml"

class CaptureType(EnumBaseStr):
    FULL = "full"
    ENTITIES_ONLY = "entities_only"
    TERRAIN_ONLY = "terrain_only"
    METRICS_ONLY = "metrics_only"

class WeatherEvent(EnumBaseStr):
    CLEAR = "clear"
    SUNNY = "sunny"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    FOG = "fog"
    DRIZZLE = "drizzle"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    LIGHT_SPRINKLE = "light_sprinkle"
    RAIN_SHOWER = "rain_shower"
    SNOW = "snow"
    BLIZZARD = "blizzard"
    HEATWAVE = "heatwave"
    DROUGHT = "drought"
    WINDY = "windy"



class Season(EnumBaseStr):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class AgentType(EnumBaseStr):
    CLIMATE_AGENT = "climate_agent"


class Agents:
    class Reinforcement(EnumBaseInt):
        NAIVE_CLIMATE = 0