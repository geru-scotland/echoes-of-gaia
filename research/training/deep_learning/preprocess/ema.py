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
import numpy as np


class EMAProcessor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def process_sequence(self, sequence):
        result = np.zeros_like(sequence)
        # TODO: IMPORTANTE, NO RESETEAR.
        # El último estado de la última secuencia, tiene que persistir
        # para que EMA pueda seguir arrastrandolo y simular como en train.

        for i in range(len(sequence)):
            if self.state is None:
                self.state = sequence[i].copy()
                result[i] = sequence[i]
            else:
                self.state = self.alpha * sequence[i] + (1 - self.alpha) * self.state
                result[i] = self.state

        return result
