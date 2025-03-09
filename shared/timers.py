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
from shared.enums.base import EnumBaseInt

class Timers:

    class System(EnumBaseInt):
        CLIMATE_UPDATE = 2
        BIOME_STATE_UPDATE = 2
        SNAPSHOT = 60
        DATA_COLLECTION = 30

    class Agents(EnumBaseInt):
        EVOLUTIONARY_CYCLE = 500
        CLIMATE_UPDATE = 2 # 2 ticks = 1 día

    class Evolutionary(EnumBaseInt):
        MUTATION_CHECK = 60
        ADAPTATION_UPDATE = 180
        SELECTION_PRESSURE = 360
        EVOLUTIONARY_CYCLE = 720

    class Calendar(EnumBaseInt):
        TICK = 1
        DAY = 2
        WEEK = 14
        MONTH = 60
        SEASON = 180
        YEAR = 720

    class Compoments:
        class Physiological(EnumBaseInt):
            AGING = 2
            HEALTH_DECAY = 2
            GROWTH = 2
            METABOLISM = 1
            ENERGY_UPDATE = 1
            TOXIN_PRODUCTION = 2
            DORMANCY_CHECK = 2

        class Environmental(EnumBaseInt):
            CLIMATE_RESPONSE = 2
            RESOURCE_ABSORPTION = 1
            STRESS_EVALUATION = 2
            WATER_BALANCE = 2

        class Ecological(EnumBaseInt):
            HERBIVORE_INTERACTION = 4
            COMPETITION = 6
            ALLELOPATHY = 8
