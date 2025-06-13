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
Energy reserves handler for entity metabolic management.

Manages energy level modifications with source tracking and bounds;
provides energy update event notifications and reserve monitoring.
Handles energy accumulation with maximum capacity constraints and
source attribution - supports energy gain-loss operations.
"""

from abc import ABC

from biome.components.handlers.base import AttributeHandler
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.events import ComponentEvent
from shared.enums.reasons import EnergyGainSource


class EnergyHandler(AttributeHandler, ABC):

    def __init__(self, event_notifier: EventNotifier, max_energy_reserves: float = 100.0):
        super().__init__(event_notifier)

        self._energy_reserves: float = round(max_energy_reserves, 2)
        self._max_energy_reserves: float = round(max_energy_reserves, 2)

    def register_events(self):
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)

    def _handle_energy_update(self, *args, **kwargs):
        new_energy: float = kwargs.get("energy_reserves", 0.0)
        self._energy_reserves = new_energy

    def modify_energy(self, energy_delta: float, source: EnergyGainSource = None) -> None:
        old_energy: float = self._energy_reserves

        self._energy_reserves = max(0.0, min(self._energy_reserves + energy_delta, self._max_energy_reserves))

        if old_energy != self._energy_reserves:
            self._event_notifier.notify(ComponentEvent.ENERGY_UPDATED,
                                        energy_reserves=self._energy_reserves,
                                        source=source)

    @property
    def energy_reserves(self) -> float:
        return self._energy_reserves

    @property
    def max_energy_reserves(self) -> float:
        return self._max_energy_reserves