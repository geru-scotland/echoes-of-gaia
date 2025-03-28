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
from typing import Dict, Any, Tuple

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType, Direction
from shared.enums.events import ComponentEvent, BiomeEvent
from biome.systems.events.event_bus import BiomeEventBus
from shared.types import Position


class MovementComponent(EntityComponent):

    def __init__(self, env: simpyEnv, event_notifier: EventNotifier,
                 movement_energy_cost: float = 1.0):
        super().__init__(env, ComponentType.MOVEMENT, event_notifier)

        self._movement_valid: bool = False
        self._movement_energy_cost: float = movement_energy_cost
        self._current_position: Position = (-1, -1)
        self._can_move: bool = True

        ComponentRegistry.get_movement_manager().register_component(id(self), self)

    def _register_events(self):
        super()._register_events()
        self._event_notifier.register(ComponentEvent.POSITION_UPDATED, self._handle_position_update)

    def _handle_position_update(self, position: Position):
        self._current_position = position

    def move(self, direction: Direction) -> bool:
        self._logger.info(f"Moving direction: {direction}")

        if direction == Direction.NONE:
            return

        if self._current_position is None:
            self._logger.warning("Cannot move: current position is None")
            return False

        y, x = self._current_position
        dy, dx = direction.value
        new_position = (y + dy, x + dx)

        self._logger.info(f"Before position: {self._current_position}")
        self._logger.info(f"After position: {new_position}")

        self._movement_valid = False

        entity_id = self._host.get_id() if self._host else -1

        BiomeEventBus.trigger(
            BiomeEvent.VALIDATE_MOVEMENT,
            entity_id=entity_id,
            new_position=new_position,
            result_callback=self.set_validation_result
        )

        if not self._movement_valid:
            self._logger.warning(f"Movement to {new_position} is not valid")
            return False

        BiomeEventBus.trigger(
            BiomeEvent.MOVE_ENTITY,
            entity_id=entity_id,
            new_position=new_position
        )

        self._current_position = new_position
        self._logger.info(f"Current position: {self._current_position}")

        return True

    def get_state(self) -> Dict[str, Any]:
        return {
            "position": self._current_position if self._current_position else (-1, -1)
        }

    def set_validation_result(self, is_valid: bool):
        self._movement_valid = is_valid

    def _handle_move_result(self, success: bool):
        return success

    def disable_notifier(self):
        super().disable_notifier()
        ComponentRegistry.get_movement_manager().unregister_component(id(self))

    @property
    def movement_energy_cost(self) -> float:
        return self._movement_energy_cost

    @property
    def current_position(self) -> Position:
        return self._current_position

    @property
    def can_move(self) -> bool:
        return self._can_move

    @movement_energy_cost.setter
    def movement_energy_cost(self, value: float) -> None:
        self._movement_energy_cost = value

    @can_move.setter
    def can_move(self, value: bool) -> None:
        self._can_move = value

    @property
    def is_active(self) -> bool:
        return self._host_alive
