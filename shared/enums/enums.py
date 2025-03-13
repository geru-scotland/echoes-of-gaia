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
from shared.enums.base import EnumBaseStr, EnumBaseInt

class BiomeType(EnumBaseStr):
    TROPICAL = "tropical"
    DESERT = "desert"
    TAIGA = "taiga"
    SAVANNA = "savanna"
    TUNDRA = "tundra"

# Maps
class TerrainType(EnumBaseInt):
    WATER_DEEP = 0
    WATER_MID = 1
    WATER_SHALLOW = 2
    SHORE = 3
    GRASS = 4
    MOUNTAIN = 5
    SNOW = 6
    SAND = 7

class ComponentType(EnumBaseStr):
    TRANSFORM = "transform"
    GROWTH = "growth"
    METABOLIC = "metabolic"
    NUTRITIONAL = "nutritional"
    STATE = "state"
    VITAL = "vital"
    CLIMATE = "climate"

class EntityState(EnumBaseStr):
    HEALTHY = "healthy"
    WEAK = "weak"
    DYING = "dying"



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
    EVOLUTION_AGENT = "evolution_agent"


class Agents:
    class Reinforcement(EnumBaseInt):
        NAIVE_CLIMATE = 0