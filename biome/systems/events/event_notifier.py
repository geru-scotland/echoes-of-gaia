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
Component-level event notification system for localized event handling.

Manages event listener registration and callback execution;
provides scoped event distribution within entity components.
Enables targeted event communication without global broadcasting.
"""

from typing import Any, Callable, Dict, List


class EventNotifier:
    def __init__(self):
        self._listeners: Dict[str, List[Any]] = {}

    def clear(self):
        self._listeners.clear()

    def register(self, event_name: str, callback: Callable):
        if event_name not in self._listeners:
            self._listeners[event_name] = []

        if callback not in self._listeners[event_name]:
            self._listeners[event_name].append(callback)

    def notify(self, event_name: str, *args, **kwargs):
        for callback in self._listeners.get(event_name, []):
            callback(*args, **kwargs)

    def unregister(self, event_name: str, callback: Any):
        if event_name in self._listeners:
            if callback in self._listeners[event_name]:
                self._listeners[event_name].remove(callback)