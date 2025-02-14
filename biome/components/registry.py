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
