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
from biome.components.physiological.growth import GrowthComponent
from biome.components.physiological.metabolic import MetabolicComponent
from biome.components.physiological.nutritional import NutritionalComponent
from biome.components.base.transform import TransformComponent
from biome.components.physiological.vital import VitalComponent

CLASS_REGISTRY = {
    "GrowthComponent": GrowthComponent,
    "VitalComponent": VitalComponent,
    "MetabolicComponent": MetabolicComponent,
    "TransformComponent": TransformComponent,
}

def register_component(name: str, cls):
    CLASS_REGISTRY[name] = cls

def get_component_class(name: str):
    return CLASS_REGISTRY.get(name)
