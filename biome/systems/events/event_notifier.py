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
from typing import Dict, List, Callable


class EventNotifier:
    def __init__(self):
        self._listeners: Dict[str, Callable] = {}

    def clear(self):
        self._listeners.clear()

    def register(self, event_name: str, callback: Callable):
        if event_name not in self._listeners:
            self._listeners[event_name] = callback

    def notify(self, event_name: str, *args, **kwargs):
        callback: Callable = self._listeners.get(event_name, None)

        if callback is not None:
            callback(*args, **kwargs)