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


class EntityProvider:
    def __init__(self, world_map: WorldMap):
        self._world_map: WorldMap = world_map
        self._entities: EntityList = self._world_map.get_entities()

    def get_entities(self) -> Tuple[EntityList, EntityList]:
        return self.get_flora(), self.get_fauna()

    def get_entities_by_type(self, entity_type: EntityType) -> EntityList:
        return [entity for entity in self._entities if entity.get_type() == entity_type]

    def get_flora(self) -> EntityList:
        return self.get_entities_by_type(EntityType.FLORA)

    def get_fauna(self) -> EntityList:
        return self.get_entities_by_type(EntityType.FAUNA)

