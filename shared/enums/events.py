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
    STAGE_CHANGE = "on_stage_change"
    SIZE_CHANGE = "on_size_change"
    MODIFIER_CHANGE = "on_modifier_change"
    STRESS_CHANGE = "on_stress_change"
    CLIMATE_RESPONSE = "on_climate_response"
    FLORA_CONSUMED = "on_flora_consumed"
    DORMANCY_TOGGLE = "on_dormancy_toggle"
    DORMANCY_REASONS_CHANGED = "on_dormancy_reasons_changed"
    DORMANCY_UPDATED = "on_dormancy_updated"
    STRESS_UPDATED = "on_stress_updated"
    ENTITY_DEATH = "entity_death"
    EXTREME_WEATHER = "extreme_weather"

class BiomeEvent(EnumBaseStr):
    CREATE_ENTITY = "create_entity"
    REMOVE_ENTITY = "remove_entity"

    CLIMATE_CHANGE = "climate_change"
    EXTREME_WEATHER = "on_extreme_weather"
    SEASON_CHANGE = "season_change"
    DISASTER = "disaster"