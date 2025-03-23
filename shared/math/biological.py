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
import math
import numpy as np

class BiologicalGrowthPatterns:

    @staticmethod
    def sigmoid_growth_curve(relative_age: float, curve_steepness: float = 6) -> float:
        min_bound = 1 / (1 + math.exp(curve_steepness / 2))
        max_bound = 1 / (1 + math.exp(-curve_steepness / 2))

        raw_growth_value = 1 / (1 + math.exp(-curve_steepness * (relative_age - 0.5)))

        return (raw_growth_value - min_bound) / (max_bound - min_bound)

    @staticmethod
    def sigmoid_growth_curve_vectorized(relative_ages: np.ndarray, curve_steepness: float = 6) -> np.ndarray:
        min_bound = 1 / (1 + np.exp(curve_steepness / 2))
        max_bound = 1 / (1 + np.exp(-curve_steepness / 2))

        raw_growth_values = 1 / (1 + np.exp(-curve_steepness * (relative_ages - 0.5)))

        return (raw_growth_values - min_bound) / (max_bound - min_bound)

    @staticmethod
    def gompertz_decay(life_proportion: float, decay_onset: float = 0.005, decay_steepness: float = 7.0) -> float:
        life_proportion = max(0, min(1, life_proportion))

        vitality_loss = 1.0 - math.exp(-decay_onset * math.exp(decay_steepness * life_proportion))

        return vitality_loss

    @staticmethod
    def von_bertalanffy_growth(relative_age: float, growth_constant: float = 2.0) -> float:
        return 1.0 - math.exp(-growth_constant * relative_age)