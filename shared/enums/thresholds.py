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

from shared.enums.base import EnumBaseFloat


class VitalThresholds:
    class Health(EnumBaseFloat):
        CRITICAL = 0.15
        LOW = 0.30
        GOOD = 0.60
        EXCELLENT = 0.80

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.02
        LOW = 0.012
        GOOD = 0.007
        EXCELLENT = 0.005


class MetabolicThresholds:
    class Energy(EnumBaseFloat):
        CRITICAL = 0.15
        LOW = 0.1
        SUFFICIENT = 0.50
        ABUNDANT = 0.80

    class StressChange(EnumBaseFloat):
        NO_ENERGY = 0.1
        CRITICAL = 0.010
        LOW = 0.008
        SUFFICIENT = -0.01
        ABUNDANT = -0.03

    class EfficiencyModifier(EnumBaseFloat):
        MIN_PHOTOSYNTHESIS = 0.4
        PHOTOSYNTHESIS_REDUCTION = 0.2
        RESPIRATION_INCREASE = 0.4


class HungerThresholds:
    class Level(EnumBaseFloat):
        CRITICAL = 0.10
        LOW = 0.30
        NORMAL = 0.70
        SATISFIED = 0.90

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.035
        LOW = 0.020
        NORMAL = 0.0
        SATISFIED = -0.010


class ThirstThresholds:
    class Level(EnumBaseFloat):
        CRITICAL = 0.10
        LOW = 0.30
        NORMAL = 0.70
        SATISFIED = 0.90

    class StressChange(EnumBaseFloat):
        CRITICAL = 0.040
        LOW = 0.025
        NORMAL = 0.0
        SATISFIED = -0.015


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
        OPTIMAL = 0.01
        HOT = 0.8
        EXTREME_HOT = 1.5
