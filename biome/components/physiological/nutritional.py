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
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import EnergyGainSource
from shared.math.constants import epsilon
from shared.timers import Timers


class NutritionalComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 nutrient_absorption_rate: float = 0.3,
                 mycorrhizal_rate: float = 0.02,
                 base_nutritive_value: float = 0.6,
                 base_toxicity: float = 0.1,
                 max_energy_reserves: float = 100.0):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.NUTRITIONAL, event_notifier, lifespan)

        self._nutrient_absorption_rate: float = nutrient_absorption_rate
        self._mycorrhizal_rate: float = mycorrhizal_rate

        self._base_nutritive_value: float = base_nutritive_value
        self._base_toxicity: float = base_toxicity
        self._current_nutritive_value: float = base_nutritive_value
        self._current_toxicity: float = base_toxicity
        self._photosynthesis_efficiency: float = 0.0

        self._stress_factor: float = self._stress_handler.stress_level / self._stress_handler.max_stress
        self._energy_factor: float = self._energy_handler.energy_reserves / self._energy_handler.max_energy_reserves

        self._logger.debug(f"NutritionalComponent initialized: Absorption rate={self._nutrient_absorption_rate}, "
                           f"Mycorrhizal efficiency={self._mycorrhizal_rate}")

        self._env.process(self._update_soil_nutrient_absorption(Timers.Compoments.Environmental.RESOURCE_ABSORPTION))
        self._env.process(self._update_mycorrhizal_activity(Timers.Compoments.Environmental.RESOURCE_ABSORPTION))

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._event_notifier.register(ComponentEvent.STRESS_UPDATED, self._handle_stress_update)
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)
        self._event_notifier.register(ComponentEvent.PHOTOSYNTHESIS_UPDATED, self._handle_photosynthesis_update)

    def _handle_stress_update(self, *args, **kwargs) -> None:
        self._stress_factor = kwargs.get("normalized_stress", 0.0)

    def _handle_energy_update(self, *args, **kwargs) -> None:
        energy_reserves: float = kwargs.get("energy_reserves", 0.0)
        self._energy_factor = energy_reserves / self._energy_handler.max_energy_reserves

    def _update_soil_nutrient_absorption(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)

        while self._host_alive:
            if not self._is_dormant:

                if self._photosynthesis_efficiency == 0.0:
                    continue

                # Por ahora, como factores que afectan la absorción:
                toxicity_factor = 1.0 - self._current_toxicity

                variability = random.uniform(0.9, 1.1)
                base_rate = self._nutrient_absorption_rate * 0.01

                absorption_rate = base_rate * self._stress_factor * (toxicity_factor * 0.7 + epsilon) * variability
                energy_gain = absorption_rate * self._energy_handler.max_energy_reserves

                self._energy_handler.modify_energy(energy_gain, source=EnergyGainSource.SOIL_NUTRIENTS)

                self._logger.debug(
                    f"[Soil nutrients] Absorbed +{energy_gain:.2f} energy "
                    f"(Factors: stress={self._stress_factor:.2f}, toxicity={toxicity_factor:.2f})"
                )

                if random.random() < 0.15:

                    toxicity_direction = random.random()

                    if toxicity_direction < (0.3 + 0.4 * self._stress_factor):
                        toxicity_change = random.uniform(0.005, 0.015) * (1.0 + self._stress_factor)
                    else:
                        toxicity_change = -random.uniform(0.003, 0.008) * (1.0 - self._stress_factor * 0.5)

                    self._current_toxicity = max(
                        self._base_toxicity,
                        min(1.0, self._current_toxicity + toxicity_change)
                    )

                    self._event_notifier.notify(
                        ComponentEvent.UPDATE_STATE,
                        NutritionalComponent,
                        toxicity=self._current_toxicity
                    )

            yield self._env.timeout(timer)

    def _update_mycorrhizal_activity(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)

        while self._host_alive:
            if not self._is_dormant:
                # Hago que las micorrizas sean más efectivas cuando la planta tiene menos energía
                mycorrhizal_bonus = self._mycorrhizal_rate * (1.0 - self._energy_factor)

                self._logger.debug(f"[Mycorrhizal activity] Generated +{mycorrhizal_bonus:.5f} energy (ratio: {self._energy_factor})")
                self._energy_handler.modify_energy(mycorrhizal_bonus, source=EnergyGainSource.MYCORRHIZAE)

            yield self._env.timeout(timer)

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
            NutritionalComponent,
            toxicity=self._current_toxicity
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "nutrient_absorption_rate": self._nutrient_absorption_rate,
            "mycorrhizal_efficiency": self._mycorrhizal_rate,
            "nutritive_value": self._current_nutritive_value,
            "toxicity": self._current_toxicity
        }

    @property
    def toxicity(self) -> float:
        return self._current_toxicity

    @property
    def nutritive_value(self) -> float:
        return self._current_nutritive_value

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
