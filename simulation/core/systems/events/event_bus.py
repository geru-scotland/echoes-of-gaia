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
from typing import Dict, List, Any


class SimulationEventBus:
    _listeners: Dict[str, List[Any]] = {}

    @classmethod
    def clear(cls):
        cls._listeners.clear()

    @classmethod
    def register(cls, event_name: str, callback: Any):
        if event_name not in cls._listeners:
            cls._listeners[event_name] = []
        cls._listeners[event_name].append(callback)

    @classmethod
    def trigger(cls, event_name, *args, **kwargs):
        for callback in cls._listeners.get(event_name, []):
            callback(*args, **kwargs)

