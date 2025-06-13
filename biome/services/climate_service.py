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
Static climate service for global climate state access.

Provides centralized access to climate state and seasonal information;
initializes with state references and callback functions.
Enables system-wide climate queries without direct dependencies.
"""

from typing import Callable, Optional

from biome.systems.climate.state import ClimateState
from shared.enums.enums import Season


class ClimateService:
    state: Optional[ClimateState] = None
    get_season: Optional[Callable]

    @classmethod
    def init_service(cls, state_ref: ClimateState, get_season_func: Callable):
        cls.state = state_ref
        cls.get_season = get_season_func

    @classmethod
    def query_state(cls) -> Optional[ClimateState]:
        return cls.state if cls.state else None

    @classmethod
    def get_current_season(cls) -> Season:
        return cls.get_season()