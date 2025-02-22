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
from typing import Tuple, Callable

from simpy import Environment as simpyEnv
from biome.components.component import EntityComponent
from shared.enums import ComponentType


class TransformComponent(EntityComponent):
    def __init__(self, env: simpyEnv = None, callback: Callable = None, x: int = -1, y: int = -1):
        super().__init__(ComponentType.TRANSFORM, env, callback)
        self.x = x
        self.y = y

    def get_position(self) -> Tuple[int, int]:
        return self.x, self.y

    def set_position(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
