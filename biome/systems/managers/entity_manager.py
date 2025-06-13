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
Entity data access provider for biome system interactions.

Provides filtered entity collections based on type and lifecycle status;
connects systems to the entity registry through a consistent interface.
Abstracts entity access and retrieval logic from world map implementation.
"""

from typing import Tuple

from biome.entities.entity import Entity
from biome.systems.maps.worldmap import WorldMap
from shared.enums.enums import EntityType
from shared.types import EntityList, EntityRegistry


class EntityProvider:
    def __init__(self, world_map: WorldMap):
        self._world_map: WorldMap = world_map
        self._entities: EntityList = self._world_map.get_entities()

    def get_entities(self, only_alive: bool = False) -> Tuple[EntityList, EntityList]:
        entities: EntityList = self._world_map.get_entities(only_alive)
        flora: EntityList = self.get_entities_by_type(entities, EntityType.FLORA)
        fauna: EntityList = self.get_entities_by_type(entities, EntityType.FAUNA)
        return flora, fauna

    def get_entities_by_type(self, entities: EntityList, entity_type: EntityType,
                             only_alive: bool = False) -> EntityList:
        if only_alive:
            return [entity for entity in entities if entity.get_type() == entity_type and entity.is_alive()]

        return [entity for entity in entities if entity.get_type() == entity_type]

    def get_flora(self, only_alive: bool = False) -> EntityList:
        entities: EntityList = self._world_map.get_entities(only_alive)
        return self.get_entities_by_type(entities, EntityType.FLORA, only_alive)

    def get_fauna(self, only_alive: bool = False) -> EntityList:
        entities: EntityList = self._world_map.get_entities(only_alive)
        return self.get_entities_by_type(entities, EntityType.FAUNA, only_alive)

    def get_entity_by_id(self, id: int) -> Entity:
        if self._world_map:
            entity_registry: EntityRegistry = self._world_map.entity_registry
            if entity_registry:
                return entity_registry.get(id)
