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
from typing import List

import numpy as np

from biome.entities.entity import Entity
from shared.types import TerrainMap, EntityRegistry, EntityIndexMap, EntityList


class WorldMap:
    def __init__(self, tile_map: TerrainMap, entity_registry: EntityRegistry, entity_index_map: EntityIndexMap):
        np.set_printoptions(threshold=np.inf)
        self._terrain_map: TerrainMap = tile_map.astype(np.int8)
        self._entity_registry: EntityRegistry  = entity_registry
        self._entity_index_map: EntityIndexMap = entity_index_map
        # mapeo ids de entidad a referencia objeto
        # y ahora, el entity map no es de index, si no de id
        print(entity_index_map)

    def get_entities(self) -> EntityList:
        entities: EntityList = []
        for _, entity in self._entity_registry.items():
            entities.append(entity)
        return entities

