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