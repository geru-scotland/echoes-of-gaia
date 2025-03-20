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
        self._entities = self._world_map.get_entities()
        return self.get_flora(), self.get_fauna()

    def get_entities_by_type(self, entity_type: EntityType, only_alive: bool = False) -> EntityList:
        self._entities = self._world_map.get_entities()

        if only_alive:
            return [entity for entity in self._entities if entity.get_type() == entity_type and entity.is_alive()]

        return [entity for entity in self._entities if entity.get_type() == entity_type]

    def get_flora(self, only_alive: bool = False) -> EntityList:
        return self.get_entities_by_type(EntityType.FLORA, only_alive)

    def get_fauna(self, only_alive: bool = False) -> EntityList:
        return self.get_entities_by_type(EntityType.FAUNA, only_alive)

