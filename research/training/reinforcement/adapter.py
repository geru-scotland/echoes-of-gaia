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

"""
Abstract adapter for reinforcement learning environment interactions.

Defines the core interface for environment adapters in the training system;
standardizes reward computation and observation collection methods. Serves
as a bridge between simulation and RL frameworks - enables consistent
environment interaction patterns across different agent types.
"""

from abc import ABC, abstractmethod


class EnvironmentAdapter(ABC):

    @abstractmethod
    def compute_reward(self, action):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self):
        raise NotImplementedError
