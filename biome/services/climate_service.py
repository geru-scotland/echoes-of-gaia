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
from typing import Optional

from biome.systems.climate.state import ClimateState


class ClimateService:
    state: Optional[ClimateState] = None

    @classmethod
    def init_service(cls, state_ref: ClimateState):
        cls.state = state_ref

    @classmethod
    def query_state(cls) -> Optional[ClimateState]:
        return cls.state if cls.state else None