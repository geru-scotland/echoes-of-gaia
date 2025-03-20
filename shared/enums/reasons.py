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
from enum import Flag, auto


class DormancyReason(Flag):
    NONE = 0
    LOW_ENERGY = auto()
    LOW_VITALITY = auto()
    ENVIRONMENTAL_STRESS = auto()


class StressReason(Flag):
    NONE = 0

    WATER_SHORTAGE = auto()
    LIGHT_DEFICIENCY = auto()
    TOXICITY = auto()
    DISEASE = auto()
    PHYSICAL_DAMAGE = auto()

    GOOD_VITALITY = auto()
    EXCELLENT_VITALITY = auto()
    LOW_VITALITY = auto()
    CRITICAL_VITALITY = auto()

    NUTRIENT_DEFICIENCY = auto()
    ENERGY_ABUNDANCE = auto()
    ENERGY_SUFFICIENT = auto()
    NO_ENERGY = auto()

    TEMPERATURE_EXTREME = auto()
    TEMPERATURE_OPTIMAL = auto()

class EnergyGainSource(Flag):
    SOIL_NUTRIENTS = auto()
    MYCORRHIZAE = auto()