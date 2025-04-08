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
from typing import NewType, Tuple, Dict, List, Optional, Literal, TypedDict
import numpy as np

EntityID = NewType('EntityID', int)
PositionTuple = Tuple[int, int]  # (y, x)
Position = NewType('Position', PositionTuple)
Distance = NewType('Distance', float)

InteractionResult = Literal["attacked", "killed", "consumed", "evaded", "none"]

EntityState = Literal["hunting", "fleeing", "wandering", "inactive"]


class EntityStats(TypedDict):
    vitality: float
    energy: float
    hunger: float
    thirst: float
    stress: float
    somatic_integrity: float
