"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from biome.components.environmental.weather_adaptation import WeatherAdaptationComponent
from biome.components.kinematics.movement import MovementComponent
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.heterotrophic_nutrition import HeterotrophicNutritionComponent
from biome.components.physiological.photosynthetic_metabolism import PhotosyntheticMetabolismComponent
from biome.components.kinematics.transform import TransformComponent
from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from biome.components.physiological.vital import VitalComponent

CLASS_REGISTRY = {
    "GrowthComponent": GrowthComponent,
    "VitalComponent": VitalComponent,
    "PhotosyntheticMetabolismComponent": PhotosyntheticMetabolismComponent,
    "TransformComponent": TransformComponent,
    "WeatherAdaptationComponent": WeatherAdaptationComponent,
    "AutotrophicNutritionComponent": AutotrophicNutritionComponent,
    "HeterotrophicNutritionComponent": HeterotrophicNutritionComponent,
    "MovementComponent": MovementComponent
}


def register_component(name: str, cls):
    CLASS_REGISTRY[name] = cls


def get_component_class(name: str):
    return CLASS_REGISTRY.get(name)
