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
        GOOD = 0.60
        EXCELLENT = 0.80

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.02
        LOW = 0.009
        GOOD = 0.004
        EXCELLENT = 0.003


class MetabolicThresholds:
    class Energy(EnumBaseFloat):
        CRITICAL = 0.15
        LOW = 0.1
        SUFFICIENT = 0.50
        ABUNDANT = 0.80

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.015
        LOW = 0.008
        SUFFICIENT = 0.005
        ABUNDANT = 0.003

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