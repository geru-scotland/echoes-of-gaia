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
from shared.base import EnumBaseInt


class Timers:
    class Agents(EnumBaseInt):
        EVOLVE = 500

    class Entity(EnumBaseInt):
        GROWTH = 25
        NUTRITIONAL_VALUE_DECAY = 5
        ENERGY = 3
        AGING = 30
        HEALTH_DECAY = 5

    class Biome(EnumBaseInt):
        CLIMATE = 50
        TEMPERATURE = 10


