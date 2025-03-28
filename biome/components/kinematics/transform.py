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
from typing import Tuple, Dict, Any

from simpy import Environment as simpyEnv
from biome.components.base.component import EntityComponent
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType


class TransformComponent(EntityComponent):
    def __init__(self, env: simpyEnv = None, event_notifier: EventNotifier = None, x: int = -1, y: int = -1):
        super().__init__(env, ComponentType.TRANSFORM, event_notifier)
        self.x = x
        self.y = y

    def _register_events(self):
        pass

    def get_position(self) -> Tuple[int, int]:
        return self.x, self.y

    def set_position(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def get_state(self) -> Dict[str, Any]:
        return {
            "position": (self.x, self.y)
        }
