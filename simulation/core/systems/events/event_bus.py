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

"""
Specialized event bus for simulation-wide communication.

Extends the base EventBus to handle simulation-specific events;
enables decoupled communication between simulation components.
Manages event listeners and triggers for biome and system events.
"""

from typing import Dict, List, Any

from shared.events.event_bus import EventBus


class SimulationEventBus(EventBus):
    _listeners: Dict[str, List[Any]]