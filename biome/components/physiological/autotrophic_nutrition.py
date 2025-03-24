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

import random
from typing import Dict, Any, Optional

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from biome.components.handlers.energy_handler import EnergyHandler
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent


class AutotrophicNutritionComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 nutrient_absorption_rate: float = 0.3,
                 mycorrhizal_rate: float = 0.02,
                 base_nutritive_value: float = 0.6,
                 base_toxicity: float = 0.1,
                 max_energy_reserves: float = 100.0):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.AUTOTROPHIC_NUTRITION, event_notifier, lifespan)

        self._nutrient_absorption_rate: float = nutrient_absorption_rate
        self._mycorrhizal_rate: float = mycorrhizal_rate

        self._base_nutritive_value: float = base_nutritive_value
        self._base_toxicity: float = base_toxicity
        self._current_nutritive_value: float = base_nutritive_value
        self._current_toxicity: float = base_toxicity
        self._photosynthesis_efficiency: float = 0.6

        self._stress_ratio: float = self._stress_handler.stress_level / self._stress_handler.max_stress
        self._energy_ratio: float = self._energy_handler.energy_reserves / self._energy_handler.max_energy_reserves

        self._logger.debug(f"AutotrophicNutritionComponent initialized: Absorption rate={self._nutrient_absorption_rate}, "
                           f"Mycorrhizal efficiency={self._mycorrhizal_rate}")

        ComponentRegistry.get_autotrophic_nutrition_manager().register_component(id(self), self)

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._energy_handler.register_events()
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)
        self._event_notifier.register(ComponentEvent.PHOTOSYNTHESIS_UPDATED, self._handle_photosynthesis_update)

    def _handle_stress_update(self, *args, **kwargs) -> None:
        self._stress_ratio = kwargs.get("normalized_stress", 0.0)

    def _handle_energy_update(self, *args, **kwargs) -> None:
        energy_reserves: float = kwargs.get("energy_reserves", 0.0)
        self._energy_ratio = energy_reserves / self._energy_handler.max_energy_reserves

    def _handle_photosynthesis_update(self, *args, **kwargs):
        old_efficiency = kwargs.get("old_efficiency", 0.0)
        new_efficiency = kwargs.get("new_efficiency", 0.0)

        self._photosynthesis_efficiency = new_efficiency

        if new_efficiency > old_efficiency:
            toxicity_change = -0.02 * (new_efficiency - old_efficiency)
            self._logger.debug(f"Photosynthesis improvement, decreasing toxicity: {toxicity_change:.4f}")
        else:
            toxicity_change = 0.04 * (old_efficiency - new_efficiency)
            self._logger.debug(f"Photosynthetic decay, increasing toxicty {toxicity_change:.4f}")

        self._current_toxicity = max(
            self._base_toxicity,
            min(1.0, self._current_toxicity + toxicity_change)
        )

        self._event_notifier.notify(
            ComponentEvent.UPDATE_STATE,
            AutotrophicNutritionComponent,
            toxicity=self._current_toxicity
        )

    def disable_notifier(self):
        super().disable_notifier()
        ComponentRegistry.get_autotrophic_nutrition_manager().unregister_component(id(self))

    def get_state(self) -> Dict[str, Any]:
        return {
            "nutrient_absorption_rate": self._nutrient_absorption_rate,
            "mycorrhizal_efficiency": self._mycorrhizal_rate,
            "nutritive_value": self._current_nutritive_value,
            "toxicity": self._current_toxicity
        }

    # Getters
    @property
    def nutrient_absorption_rate(self) -> float:
        return self._nutrient_absorption_rate

    @property
    def mycorrhizal_rate(self) -> float:
        return self._mycorrhizal_rate

    @property
    def base_nutritive_value(self) -> float:
        return self._base_nutritive_value

    @property
    def base_toxicity(self) -> float:
        return self._base_toxicity

    @property
    def current_nutritive_value(self) -> float:
        return self._current_nutritive_value

    @property
    def current_toxicity(self) -> float:
        return self._current_toxicity

    @property
    def photosynthesis_efficiency(self) -> float:
        return self._photosynthesis_efficiency

    @property
    def stress_ratio(self) -> float:
        return self._stress_ratio

    @property
    def energy_ratio(self) -> float:
        return self._energy_ratio

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
    def lifespan(self) -> float:
        return self._lifespan

    @property
    def is_active(self) -> bool:
        return self._host_alive

    @property
    def is_dormant(self) -> bool:
        return self._is_dormant

    # Setters
    @nutrient_absorption_rate.setter
    def nutrient_absorption_rate(self, value: float) -> None:
        self._nutrient_absorption_rate = value

    @mycorrhizal_rate.setter
    def mycorrhizal_rate(self, value: float) -> None:
        self._mycorrhizal_rate = value

    @base_nutritive_value.setter
    def base_nutritive_value(self, value: float) -> None:
        self._base_nutritive_value = value

    @base_toxicity.setter
    def base_toxicity(self, value: float) -> None:
        self._base_toxicity = value

    @current_nutritive_value.setter
    def current_nutritive_value(self, value: float) -> None:
        self._current_nutritive_value = value

    @current_toxicity.setter
    def current_toxicity(self, value: float) -> None:
        self._current_toxicity = value

    @photosynthesis_efficiency.setter
    def photosynthesis_efficiency(self, value: float) -> None:
        self._photosynthesis_efficiency = value

    @stress_ratio.setter
    def stress_ratio(self, value: float) -> None:
        self._stress_ratio = value

    @energy_ratio.setter
    def energy_ratio(self, value: float) -> None:
        self._energy_ratio = value