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
Heterotrophic nutrition component for fauna energy metabolism.

Manages hunger, thirst and energy reserve dynamics for fauna entities;
handles food and water consumption with metabolism efficiency modifiers.
Integrates with stress system for metabolic rate adjustments - provides
comprehensive physiological state tracking for fauna survival processes.
"""

from typing import Any, Dict

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.components.handlers.energy_handler import EnergyHandler
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent


class HeterotrophicNutritionComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 max_energy_reserves: float = 100.0,
                 hunger_rate: float = 0.05,
                 thirst_rate: float = 0.08,
                 metabolism_efficiency: float = 1.0):
        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.HETEROTROPHIC_NUTRITION, event_notifier, lifespan)

        self._hunger_rate: float = hunger_rate
        self._thirst_rate: float = thirst_rate

        self._hunger_level: float = 100.0
        self._thirst_level: float = 100.0
        self._max_level: float = 100.0

        self._metabolism_efficiency: float = metabolism_efficiency

        self._logger.debug(
            f"HeterotrophicNutritionComponent initialized: Energy={self._energy_handler.energy_reserves}/{max_energy_reserves}, "
            f"Hunger={self._hunger_level}, Thirst={self._thirst_level}")

        ComponentRegistry.get_heterotrophic_nutrition_manager().register_component(id(self), self)

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._energy_handler.register_events()
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)

    def _handle_stress_update(self, *args, **kwargs):
        stress_level = kwargs.get("stress_level", 0.0)
        max_stress = self._stress_handler.max_stress

        stress_factor = 1.0 + (stress_level / max_stress) * 0.5
        self._effective_hunger_rate = self._hunger_rate * stress_factor
        self._effective_thirst_rate = self._thirst_rate * stress_factor

    def consume_food(self, nutrition_value: float) -> None:
        energy_gain = nutrition_value * self._metabolism_efficiency
        hunger_factor = 5.0 + (100.0 - self._hunger_level) * 0.25
        amplified_nutrition = nutrition_value * hunger_factor
        hunger_reduction = min(self._max_level - self._hunger_level, amplified_nutrition)

        self._hunger_level = min(self._max_level, self._hunger_level + hunger_reduction)
        energy_gain = amplified_nutrition * self._metabolism_efficiency
        self._energy_handler.modify_energy(energy_gain)

        self._event_notifier.notify(
            ComponentEvent.UPDATE_STATE,
            HeterotrophicNutritionComponent,
            hunger_level=self._hunger_level
        )

    def consume_water(self, hydration_value: float) -> None:
        thirst_reduction = min(self._max_level - self._thirst_level, hydration_value)
        self._thirst_level = min(self._max_level, self._thirst_level + thirst_reduction)

        energy_boost = hydration_value * 0.3
        self._energy_handler.modify_energy(energy_boost)

        self._event_notifier.notify(
            ComponentEvent.UPDATE_STATE,
            HeterotrophicNutritionComponent,
            thirst_level=self._thirst_level
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "energy_reserves": self._energy_handler.energy_reserves,
            "max_energy_reserves": self._energy_handler.max_energy_reserves,
            "hunger_level": self._hunger_level,
            "thirst_level": self._thirst_level,
            "metabolism_efficiency": self._metabolism_efficiency,
            "hunger_rate": self._hunger_rate,
            "thirst_rate": self._thirst_rate
        }

    def disable_notifier(self):
        super().disable_notifier()
        ComponentRegistry.get_heterotrophic_nutrition_manager().unregister_component(id(self))

    # Getters
    @property
    def hunger_level(self) -> float:
        return self._hunger_level

    @property
    def thirst_level(self) -> float:
        return self._thirst_level

    @property
    def hunger_rate(self) -> float:
        return self._hunger_rate

    @property
    def thirst_rate(self) -> float:
        return self._thirst_rate

    @property
    def metabolism_efficiency(self) -> float:
        return self._metabolism_efficiency

    @property
    def energy_reserves(self) -> float:
        return self._energy_handler.energy_reserves

    @property
    def max_energy_reserves(self) -> float:
        return self._energy_handler.max_energy_reserves

    @property
    def stress_handler(self) -> StressHandler:
        return self._stress_handler

    @property
    def energy_handler(self) -> EnergyHandler:
        return self._energy_handler

    @property
    def event_notifier(self) -> EventNotifier:
        return self._event_notifier

    @property
    def is_active(self) -> bool:
        return self._host_alive

    # Setters
    @hunger_level.setter
    def hunger_level(self, value: float) -> None:
        self._hunger_level = value

    @thirst_level.setter
    def thirst_level(self, value: float) -> None:
        self._thirst_level = value

    @hunger_rate.setter
    def hunger_rate(self, value: float) -> None:
        self._hunger_rate = value

    @thirst_rate.setter
    def thirst_rate(self, value: float) -> None:
        self._thirst_rate = value

    @metabolism_efficiency.setter
    def metabolism_efficiency(self, value: float) -> None:
        self._metabolism_efficiency = value
