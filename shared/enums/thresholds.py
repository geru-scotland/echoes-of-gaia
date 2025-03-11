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

from shared.enums.base import EnumBaseStr, EnumBaseFloat


class VitalThresholds:
    class Health(EnumBaseFloat):
        CRITICAL = 0.15
        LOW = 0.30
        GOOD = 0.70
        EXCELLENT = 0.90

    class StressChange(EnumBaseFloat):
        CRITICAL = 1.0
        LOW = 0.5
        GOOD = -0.9
        EXCELLENT = -1.2


class MetabolicThresholds:
    class Energy(EnumBaseFloat):
        CRITICAL = 0.15
        LOW = 0.30
        SUFFICIENT = 0.50
        ABUNDANT = 0.80

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.8
        LOW = 0.6
        SUFFICIENT = -0.6
        ABUNDANT = -0.9

    class EfficiencyModifier(EnumBaseFloat):
        MIN_PHOTOSYNTHESIS = 0.4
        PHOTOSYNTHESIS_REDUCTION = 0.2
        RESPIRATION_INCREASE = 0.4


class ClimateThresholds:
    class Temperature(EnumBaseFloat):
        EXTREME_COLD = -10.0
        COLD = 5.0
        OPTIMAL_LOW = 15.0
        OPTIMAL_HIGH = 30.0
        HOT = 35.0
        EXTREME_HOT = 40.0

    class StressChange(EnumBaseFloat):
        EXTREME_COLD = 1.5
        COLD = 0.8
        OPTIMAL = -0.1
        HOT = 0.8
        EXTREME_HOT = 1.5