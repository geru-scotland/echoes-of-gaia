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
from biome.components.entities.growth import GrowthComponent
from biome.components.entities.nutritional import NutritionalValueComponent

CLASS_REGISTRY = {
    "GrowthComponent": GrowthComponent,
    "NutritionalValueComponent": NutritionalValueComponent,
}

def register_component(name: str, cls):
    CLASS_REGISTRY[name] = cls

def get_component_class(name: str):
    return CLASS_REGISTRY.get(name)
