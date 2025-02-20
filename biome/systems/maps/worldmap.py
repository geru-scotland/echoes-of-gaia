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
import numpy as np

from shared.types import TileMap, BiomeStoreData, EntityList


class WorldMap:
    def __init__(self, tile_map: TileMap, entities: EntityList):
        self._terrain_layer = tile_map
        print(tile_map)
        print(len(entities))
        self._entity_layer = np.zeros(self._terrain_layer.shape)



