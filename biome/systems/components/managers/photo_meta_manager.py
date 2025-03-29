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
from logging import Logger

import numpy as np
from simpy import Environment as simpyEnv

from biome.components.physiological.photosynthetic_metabolism import PhotosyntheticMetabolismComponent
from biome.systems.components.managers.base import BaseComponentManager
from shared.enums.events import ComponentEvent
from shared.enums.reasons import StressReason
from shared.enums.strings import Loggers
from shared.enums.thresholds import MetabolicThresholds
from shared.timers import Timers
from utils.loggers import LoggerManager


class PhotosyntheticMetabolismComponentManager(BaseComponentManager[PhotosyntheticMetabolismComponent]):
    def __init__(self, env: simpyEnv):
        super().__init__(env)

        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._env.process(self._update_all_metabolism(Timers.Components.Physiological.METABOLISM))
        self._env.process(self._synchronize_photosynthesis_efficiency(Timers.Components.Physiological.STRESS_UPDATE))
        self._env.process(self._update_metabolic_stress(Timers.Components.Physiological.STRESS_UPDATE))

    def _update_metabolic_stress(self, timer: int):
        yield self._env.timeout(timer)
        while True:
            active_components = self._get_active_components()

            if active_components:
                for component in active_components:
                    energy = component.energy_factor

                    if energy <= 0.0005:
                        stress_change = MetabolicThresholds.StressChange.NO_ENERGY
                        stress_reason = StressReason.NO_ENERGY

                    elif 0.0 < energy < MetabolicThresholds.Energy.CRITICAL:
                        stress_change = MetabolicThresholds.StressChange.CRITICAL
                        stress_reason = StressReason.NUTRIENT_DEFICIENCY

                    elif energy < MetabolicThresholds.Energy.LOW:
                        stress_change = MetabolicThresholds.StressChange.LOW
                        stress_reason = StressReason.NUTRIENT_DEFICIENCY

                    elif energy > MetabolicThresholds.Energy.ABUNDANT:
                        stress_change = MetabolicThresholds.StressChange.ABUNDANT
                        stress_reason = StressReason.ENERGY_ABUNDANCE

                    elif energy > MetabolicThresholds.Energy.SUFFICIENT:
                        stress_change = MetabolicThresholds.StressChange.SUFFICIENT
                        stress_reason = StressReason.ENERGY_SUFFICIENT

                    else:
                        stress_change = None
                        stress_reason = None

                    if stress_change is not None and stress_reason is not None:
                        component.stress_handler.modify_stress(stress_change, stress_reason)

            yield self._env.timeout(timer)

    def _update_all_metabolism(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            active_components = self._get_active_components()

            if active_components:
                photosynthesis_efficiencies = np.array([comp.photosynthesis_efficiency for comp in active_components])
                light_availabilities = np.array([comp.light_availability for comp in active_components])
                temperature_modifiers = np.array([comp.temperature_modifier for comp in active_components])
                water_modifiers = np.array([comp.water_modifier for comp in active_components])
                metabolic_activities = np.array([comp.metabolic_activity for comp in active_components])
                respiration_rates = np.array([comp.respiration_rate for comp in active_components])
                # max_energy_reserves = np.array([comp._energy_handler.max_energy_reserves for comp in active_components])

                dormancy_factors = np.array([0.15 if comp.is_dormant else 1.0 for comp in active_components])

                effective_photosynthesis = (
                        photosynthesis_efficiencies *
                        light_availabilities *
                        temperature_modifiers *
                        water_modifiers *
                        metabolic_activities *
                        dormancy_factors
                )

                effective_respiration = (
                        respiration_rates *
                        metabolic_activities *
                        np.array([0.1 if comp.is_dormant else 1.0 for comp in active_components])
                )

                optimal_respiration_ratios = np.array([comp.optimal_respiration_ratio for comp in active_components])
                current_ratios = effective_respiration / np.maximum(0.001, effective_photosynthesis)
                ratio_differences = np.abs(current_ratios - optimal_respiration_ratios)

                metabolic_efficiencies = np.maximum(0.1, 1.0 - (ratio_differences / optimal_respiration_ratios) * 1.5)

                base_energy_changes = (effective_photosynthesis - effective_respiration)
                base_energy_changes *= metabolic_efficiencies

                entropy_decays = np.full_like(base_energy_changes, 0.02)

                biological_ages = np.array([comp.biological_age if comp.biological_age is not None else 0.0
                                            for comp in active_components])
                lifespans = np.array([comp.lifespan if comp.lifespan is not None else 1.0
                                      for comp in active_components])

                current_energy_reserves = np.array([comp.energy_handler.energy_reserves for comp in active_components])

                relative_ages = biological_ages / (lifespans * float(Timers.Calendar.YEAR))
                age_factors = 1.0 + (relative_ages ** 2) * 0.5
                age_decays = np.maximum(0.01, current_energy_reserves * 0.002 * age_factors)

                final_energy_changes = base_energy_changes - (entropy_decays + age_decays)

                for i, component in enumerate(active_components):
                    self._logger.debug(
                        f"[Metabolic Update | DEBUG:Tick={self._env.now}] "
                        f"Stress level={component.stress_handler.stress_level:4f}]"
                        f"Energy: {component.energy_handler.energy_reserves:.3f}/{component.energy_handler.max_energy_reserves:.3f}, "
                        f"Photosynthesis: Eff={component.photosynthesis_efficiency:.3f}, "
                        f"Light={component.light_availability:.3f}, "
                        f"Temp Mod={component.temperature_modifier:.3f}, "
                        f"Water Mod={component.water_modifier:.3f}, "
                        f"Activity={component.metabolic_activity:.3f}, "
                        f"Respiration: Rate={component.respiration_rate:.3f}, "
                        f"Energy Change={final_energy_changes[i]:.3f}, "
                    )

                    component.energy_handler.modify_energy(final_energy_changes[i])

            yield self._env.timeout(timer)

    def _synchronize_photosynthesis_efficiency(self, timer: int):
        yield self._env.timeout(timer)

        while True:
            self.handle_all_stress_updates()
            yield self._env.timeout(timer)

    def handle_all_stress_updates(self):
        active_components = self._get_active_components()

        if active_components:
            normalized_stresses = np.array([
                comp.stress_handler.stress_level / comp.stress_handler.max_stress
                for comp in active_components
            ])

            # old_efficiencies = np.array([
            #     comp._photosynthesis_efficiency for comp in active_components
            # ])

            base_efficiencies = np.array([
                comp.base_photosynthesis_efficiency for comp in active_components
            ])

            new_efficiencies = np.zeros_like(normalized_stresses)

            low_stress_mask = normalized_stresses < 0.4

            if np.any(low_stress_mask):
                low_stress = normalized_stresses[low_stress_mask]
                adaptation_factors = 1.0 + 0.25 * (1.0 - (low_stress - 0.25) ** 2 * 16)
                adapted_efficiencies = np.minimum(
                    base_efficiencies[low_stress_mask] * 1.5,
                    base_efficiencies[low_stress_mask] * adaptation_factors
                )
                delta_adaptations = adapted_efficiencies - base_efficiencies[low_stress_mask]
                new_efficiencies[low_stress_mask] = base_efficiencies[low_stress_mask] + delta_adaptations

            stress_impact_photosynthesis = np.exp(-1.5 * normalized_stresses)
            non_adapted_efficiencies = np.maximum(
                0.25,
                base_efficiencies * stress_impact_photosynthesis
            )

            delta_non_adapted = non_adapted_efficiencies - base_efficiencies
            new_efficiencies += delta_non_adapted

            new_efficiencies = np.maximum(
                0.25,
                np.minimum(base_efficiencies * 1.5, new_efficiencies)
            )

            stress_impact_respiration = 1 + (1.0 * normalized_stresses ** 2)
            new_respiration_rates = np.minimum(
                0.2,
                np.array([comp.base_respiration_rate for comp in active_components]) * stress_impact_respiration
            )

            for i, component in enumerate(active_components):
                if component.photosynthesis_efficiency != new_efficiencies[i]:
                    old_efficiency = component.photosynthesis_efficiency
                    component._photosynthesis_efficiency = new_efficiencies[i]
                    component._respiration_rate = new_respiration_rates[i]

                    component.event_notifier.notify(
                        ComponentEvent.PHOTOSYNTHESIS_UPDATED,
                        old_efficiency=old_efficiency,
                        new_efficiency=component.photosynthesis_efficiency,
                    )

                    self._logger.debug(
                        f"Stress Update Received: Norm.Stress={normalized_stresses[i]:.2f}, "
                        f"Photosynthesis={component.photosynthesis_efficiency:.2f}, "
                        f"Respiration={component.respiration_rate:.2f}"
                    )
