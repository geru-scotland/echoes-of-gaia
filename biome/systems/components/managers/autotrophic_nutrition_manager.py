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
from typing import Dict, Any, List

import numpy as np
from simpy import Environment as simpyEnv

from biome.components.physiological.autotrophic_nutrition import AutotrophicNutritionComponent
from shared.enums.events import ComponentEvent
from shared.enums.reasons import EnergyGainSource
from shared.math.constants import epsilon
from shared.timers import Timers


class AutotrophicNutritionComponentManager:
    def __init__(self, env: simpyEnv):
        self._env = env
        self._components: Dict[int, Any] = {}
        self._component_ids = set()

        self._soil_nutrient_process = self._env.process(
            self._update_all_soil_nutrient_absorption(Timers.Compoments.Environmental.RESOURCE_ABSORPTION))

        self._mycorrhizal_process = self._env.process(
            self._update_all_mycorrhizal_activity(Timers.Compoments.Environmental.RESOURCE_ABSORPTION))

    def register_component(self, component_id: int, component: Any) -> None:
        self._components[component_id] = component
        self._component_ids.add(component_id)

    def unregister_component(self, component_id: int) -> None:
        if component_id in self._components:
            del self._components[component_id]
            self._component_ids.discard(component_id)

    def _get_active_components(self) -> List[Any]:
        return [comp for comp in self._components.values() if comp._host_alive]

    def _get_non_dormant_components(self) -> List[Any]:
        return [comp for comp in self._components.values()
                if comp._host_alive and not comp._is_dormant]

    def _update_all_soil_nutrient_absorption(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_non_dormant_components()

            if active_components:
                photosynthesis_efficiencies = np.array([comp._photosynthesis_efficiency for comp in active_components])
                active_photosynthesis_mask = photosynthesis_efficiencies > 0.0

                if np.any(active_photosynthesis_mask):
                    active_phot_components = [comp for i, comp in enumerate(active_components)
                                              if active_photosynthesis_mask[i]]

                    stress_ratios = np.array([1 - comp._stress_handler.stress_level / comp._stress_handler.max_stress
                                              for comp in active_phot_components])

                    toxicities = np.array([comp._current_toxicity for comp in active_phot_components])
                    toxicity_factors = 1.0 - toxicities

                    variabilities = np.random.uniform(0.9, 1.1, len(active_phot_components))

                    base_rates = np.array([comp._nutrient_absorption_rate * 0.01 for comp in active_phot_components])

                    absorption_rates = base_rates * stress_ratios * (toxicity_factors * 0.7 + epsilon) * variabilities

                    energy_gains = absorption_rates * np.array([comp._energy_handler.max_energy_reserves
                                                                for comp in active_phot_components])

                    toxicity_change_flags = np.random.random(len(active_phot_components)) < 0.15

                    if np.any(toxicity_change_flags):
                        toxicity_change_components = [comp for i, comp in enumerate(active_phot_components)
                                                      if toxicity_change_flags[i]]

                        toxicity_change_indices = np.where(toxicity_change_flags)[0]

                        toxicity_directions = np.random.random(len(toxicity_change_components))
                        toxicity_stress_thresholds = 0.3 + 0.4 * np.array([
                            comp._stress_handler.stress_level / comp._stress_handler.max_stress
                            for comp in toxicity_change_components
                        ])

                        increase_toxicity_mask = toxicity_directions < toxicity_stress_thresholds
                        decrease_toxicity_mask = ~increase_toxicity_mask

                        toxicity_changes = np.zeros(len(toxicity_change_components))

                        if np.any(increase_toxicity_mask):
                            stress_ratios_for_increase = np.array([
                                comp._stress_handler.stress_level / comp._stress_handler.max_stress
                                for i, comp in enumerate(toxicity_change_components)
                                if increase_toxicity_mask[i]
                            ])
                            increase_values = np.random.uniform(0.005, 0.015, np.sum(increase_toxicity_mask)) * (
                                    1.0 + stress_ratios_for_increase)
                            toxicity_changes[increase_toxicity_mask] = increase_values

                        if np.any(decrease_toxicity_mask):
                            stress_ratios_for_decrease = np.array([
                                1.0 - comp._stress_handler.stress_level / comp._stress_handler.max_stress
                                for i, comp in enumerate(toxicity_change_components)
                                if decrease_toxicity_mask[i]
                            ])
                            decrease_values = -np.random.uniform(0.003, 0.008, np.sum(
                                decrease_toxicity_mask)) * stress_ratios_for_decrease
                            toxicity_changes[decrease_toxicity_mask] = decrease_values

                    for i, component in enumerate(active_phot_components):
                        component._energy_handler.modify_energy(energy_gains[i], source=EnergyGainSource.SOIL_NUTRIENTS)

                        component._logger.debug(
                            f"[Soil nutrients] Absorbed +{energy_gains[i]:.2f} energy "
                            f"(Factors: stress={stress_ratios[i]:.2f}, toxicity={toxicity_factors[i]:.2f})"
                        )

                        if toxicity_change_flags[i]:
                            index_in_change_array = np.where(toxicity_change_indices == i)[0][0]
                            component._current_toxicity = max(
                                component._base_toxicity,
                                min(1.0, component._current_toxicity + toxicity_changes[index_in_change_array])
                            )

                            component._event_notifier.notify(
                                ComponentEvent.UPDATE_STATE,
                                AutotrophicNutritionComponent,
                                toxicity=component._current_toxicity
                            )

            yield self._env.timeout(timer)

    def _update_all_mycorrhizal_activity(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_non_dormant_components()

            if active_components:
                energy_ratios = np.array([
                    comp._energy_handler.energy_reserves / comp._energy_handler.max_energy_reserves
                    for comp in active_components
                ])

                energy_factors = 1.0 - energy_ratios

                mycorrhizal_rates = np.array([comp._mycorrhizal_rate for comp in active_components])
                mycorrhizal_bonuses = mycorrhizal_rates * energy_factors

                for i, component in enumerate(active_components):
                    component._logger.debug(
                        f"[Mycorrhizal activity] Generated +{mycorrhizal_bonuses[i]:.5f} energy (ratio: {energy_factors[i]})"
                    )
                    component._energy_handler.modify_energy(mycorrhizal_bonuses[i], source=EnergyGainSource.MYCORRHIZAE)

            yield self._env.timeout(timer)