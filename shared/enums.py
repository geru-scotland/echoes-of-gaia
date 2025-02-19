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
    NUTRITIONAL = "nutritional"
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
    OAK_TREE = "oak_tree"
    BRAMBLE = "bramble"
    MUSHROOM = "mushroom"

class FaunaType(EnumBaseStr):
    DEER = "deer"
    BOAR = "boar"
    FOX = "fox"

class Timers:
    class Agents(EnumBaseInt):
        EVOLVE = 500

    class Entity(EnumBaseInt):
        GROWTH = 25
        NUTRITIONAL_VALUE_DECAY = 5
        ENERGY = 3

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