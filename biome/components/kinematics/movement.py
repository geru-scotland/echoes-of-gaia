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
Entity movement component with direction-based position calculation.

Manages directional movement with energy cost tracking and validation;
handles position updates through centralized event notification.
Provides movement capability control and position calculation methods;
coordinates with validation systems for terrain-aware traversal.
"""

from typing import Dict, Any, Tuple

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType, Direction, PositionNotValidReason
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

    def calculate_new_position(self, direction: Direction, previous_position: Position = None) -> Position:
        y, x = previous_position if previous_position else self._current_position
        dy, dx = direction.value
        new_position = (y + dy, x + dx)
        return new_position

    def move(self, direction: Direction) -> bool:
        self._logger.debug(f"Moving direction: {direction}")

        heterotrophic_component = self._host.get_component(ComponentType.HETEROTROPHIC_NUTRITION)
        if heterotrophic_component and heterotrophic_component.energy_reserves <= 0:
            self._logger.debug("Cannot move: energy depleted")
            return False
        if self._current_position is None:
            self._logger.warning("Cannot move: current position is None")
            return False

        new_position = self.calculate_new_position(direction)

        self._logger.debug(f"Before position: {self._current_position}")
        self._logger.debug(f"After position: {new_position}")

        self._movement_valid = False

        entity_id = self._host.get_id() if self._host else -1

        validation_result = [False]
        validation_reason = [None]

        def validation_callback(is_valid: bool, reason: PositionNotValidReason = PositionNotValidReason.NONE):
            validation_result[0] = is_valid
            validation_reason[0] = reason

        BiomeEventBus.trigger(
            BiomeEvent.VALIDATE_MOVEMENT,
            entity_id=self._host.get_id(),
            new_position=new_position,
            result_callback=validation_callback
        )

        if not validation_result[0][0]:
            self._logger.debug(f"Invalid movement")
            return False

        BiomeEventBus.trigger(
            BiomeEvent.MOVE_ENTITY,
            entity_id=entity_id,
            new_position=new_position
        )

        self._current_position = new_position
        self._logger.debug(f"Current position: {self._current_position}")

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
