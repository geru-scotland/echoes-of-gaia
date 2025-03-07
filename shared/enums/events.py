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
from shared.enums.base import EnumBaseStr


class ComponentEvent(EnumBaseStr):
    UPDATE_STATE = "on_component_update"

class BiomeEvent(EnumBaseStr):
    CREATE_ENTITY = "create_entity"
    REMOVE_ENTITY = "remove_entity"

    CLIMATE_CHANGE = "climate_change"
    SEASON_CHANGE = "season_change"
    DISASTER = "disaster"