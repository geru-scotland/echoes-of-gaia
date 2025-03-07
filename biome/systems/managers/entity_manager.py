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
from typing import Tuple

from biome.systems.maps.worldmap import WorldMap
from shared.enums.enums import EntityType
from shared.types import EntityList


class EntityManager:
    def __init__(self, world_map: WorldMap):
        self.world_map: WorldMap = world_map

    def get_entities(self) -> Tuple[EntityList, EntityList]:
        entities = self.world_map.get_entities()
        return self.get_flora(entities), self.get_fauna(entities)

    def get_entities_by_type(self, entity_type: EntityType,
                             entities: EntityList) -> EntityList:
        return [entity for entity in entities if entity.get_type() == entity_type]

    def get_flora(self, entities: EntityList) -> EntityList:
        return self.get_entities_by_type(EntityType.FLORA, entities)

    def get_fauna(self, entities: EntityList) -> EntityList:
        return self.get_entities_by_type(EntityType.FAUNA, entities)

