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
    _env: Optional[simpyEnv] = None
    _growth_manager = None
    _vital_manager = None

    @classmethod
    def initialize(cls, env: simpyEnv, cleanup_dead_entities: bool = False):
        cls._env = env
        # Importo aqui para evitar dep. circulares. TODO: Revisar.
        from biome.systems.components.managers.growth_manager import GrowthComponentManager
        from biome.systems.components.managers.vital_manager import VitalComponentManager
        from biome.systems.components.managers.autotrophic_nutrition_manager import AutotrophicNutritionComponentManager
        from biome.systems.components.managers.photo_meta_manager import PhotosyntheticMetabolismComponentManager

        cls._growth_manager = GrowthComponentManager(env)
        cls._vital_manager = VitalComponentManager(env, cleanup_dead_entities)
        cls._photosynthetic_metabolic_manager = PhotosyntheticMetabolismComponentManager(env)
        cls._autotrophic_nutrition_manager = AutotrophicNutritionComponentManager(env)

    @classmethod
    def get_growth_manager(cls):
        if cls._growth_manager is None:
            raise RuntimeError("GrowthComponentManager hasn't been initialized")
        return cls._growth_manager

    @classmethod
    def get_vital_manager(cls):
        if cls._vital_manager is None:
            raise RuntimeError("VitalComponentManager  hasn't been initialized")
        return cls._vital_manager

    @classmethod
    def get_photosynthetic_metabolism_manager (cls):
        if cls._photosynthetic_metabolic_manager is None:
            raise RuntimeError("VitalComponentManager hasn't been initialized")
        return cls._photosynthetic_metabolic_manager


    @classmethod
    def get_autotrophic_nutrition_manager(cls):
        if cls._autotrophic_nutrition_manager is None:
            raise RuntimeError("AutotrophicNutritionManager hasn't been initialized")
        return cls._autotrophic_nutrition_manager

