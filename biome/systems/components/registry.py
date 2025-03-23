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
from typing import Dict, Optional
from simpy import Environment as simpyEnv



class ComponentRegistry:
    _growth_manager = None

    @classmethod
    def initialize(cls, env: simpyEnv):
        from biome.systems.components.managers.growth_manager import GrowthComponentManager
        cls._growth_manager = GrowthComponentManager(env)

    @classmethod
    def get_growth_manager(cls):
        if cls._growth_manager is None:
            raise RuntimeError("GrowthComponentManager no ha sido inicializado")
        return cls._growth_manager
