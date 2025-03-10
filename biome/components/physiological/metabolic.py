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
from typing import Optional

from biome.components.base.component import EntityComponent, FloraComponent
from simpy import Environment as simpyEnv

from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason
from shared.timers import Timers


class MetabolicComponent(FloraComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier,
                 photosynthesis_efficiency: float = 0.75, respiration_rate: float = 0.05,
                 metabolic_activity: float = 1.0, energy_reserves: float = 100.0, max_energy_reserves: float = 100.0):

        super().__init__(env, ComponentType.METABOLIC, event_notifier)
        self._photosynthesis_efficiency: float = round(photosynthesis_efficiency, 2)
        self._respiration_rate: float = round(respiration_rate, 2)
        self._metabolic_activity: float = round(metabolic_activity, 2)
        self._energy_reserves: float = round(energy_reserves, 2)
        self._max_energy_reserves: float = round(max_energy_reserves, 2)

        # TODO: Para light availability, hace falta más información en el state
        # quizá agregar el weather effect y analizar desde aquí, mediante ClimateService
        self._light_availability: float = 1.0
        self._temperature_modifier: float = 1.0
        self._water_modifier: float = 1.0

        self._logger.debug(f"Metabolic component initialized: Photosynthesis={self._photosynthesis_efficiency}, "
                           f"Respiration={self._respiration_rate}, Energy={self._energy_reserves}")

        self._env.process(self._update_metabolism(Timers.Compoments.Physiological.METABOLISM))

    def _register_events(self):
        super()._register_events()

    def _update_metabolism(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            # Hago que aunque en dormancia, aún haga photositensis y respire
            # aunque bajo minimos - pero así, puede recuperar algo de energia
            # y salir de dormancia, que antes se me quedaba siempre dormidica.
            photosynthesis_factor: float = 1.0 if not self._is_dormant else 0.15

            effective_photosynthesis: float = (
                    self._photosynthesis_efficiency *
                    self._light_availability *
                    self._temperature_modifier *
                    self._water_modifier *
                    self._metabolic_activity *
                    photosynthesis_factor
            )

            # Durante dormancia, reduzco respiradción drásticamente
            respiration_factor: float = 1.0 if not self._is_dormant else 0.1
            effective_respiration: float = self._respiration_rate * self._metabolic_activity * respiration_factor

            # fotosintesis - anabolismo, produce
            # respiración, catabolismo, quema
            energy_change: float = effective_photosynthesis - effective_respiration
            self._energy_reserves += energy_change

            # Clamp
            self._energy_reserves = max(0.0, min(self._energy_reserves, self._max_energy_reserves))

            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, MetabolicComponent,
                                        energy_reserves=self._energy_reserves)

            energy_threshold: float = self._max_energy_reserves * 0.2

            if not self._is_dormant and self._energy_reserves < energy_threshold:
                self.request_dormancy(DormancyReason.LOW_ENERGY, True)
            elif self._is_dormant and self._energy_reserves > energy_threshold:
                self.request_dormancy(DormancyReason.LOW_ENERGY, False)

            base_generation = self._photosynthesis_efficiency
            total_modifiers = (self._light_availability * self._temperature_modifier *
                               self._water_modifier * self._metabolic_activity * photosynthesis_factor)

            self._logger.info(
                f"Energy generation breakdown: Base={base_generation:.2f}, "
                f"Modifiers={total_modifiers:.2f}, Total={effective_photosynthesis:.2f}"
            )

            if effective_photosynthesis <= effective_respiration:
                self._logger.warning(
                    f"Energy deficit! Photosynthesis={effective_photosynthesis:.2f} <= "
                    f"Respiration={effective_respiration:.2f}"
                )
            self._logger.info(
                f"[Metabolic Update] [Tick: {self._env.now}] "
                f"Energy: {self._energy_reserves:.3f}/{self._max_energy_reserves:.3f}, "
                f"Photosynthesis: Eff={self._photosynthesis_efficiency:.3f}, "
                f"Light={self._light_availability:.3f}, "
                f"Temp Mod={self._temperature_modifier:.3f}, "
                f"Water Mod={self._water_modifier:.3f}, "
                f"Activity={self._metabolic_activity:.3f}, "
                f"Effective Photo={effective_photosynthesis:.3f}, "
                f"Respiration: Rate={self._respiration_rate:.3f}, "
                f"Effective Resp={effective_respiration:.3f}, "
                f"Energy Change={energy_change:.3f}, "
                f"Dormant={self._is_dormant}"
            )
            yield self._env.timeout(timer)

    def set_environmental_modifiers(self, light: float = None, temperature: float = None, water: float = None):
        if light is not None:
            self._light_availability = max(0.0, min(1.0, light))

        if temperature is not None:
            self._temperature_modifier = max(0.0, min(1.5, temperature))

        if water is not None:
            self._water_modifier = max(0.0, min(1.0, water))

    def set_metabolic_activity(self, activity_level: float):
        self._metabolic_activity = max(0.1, min(1.0, activity_level))

    def consume_energy(self, amount: float) -> bool:
        if amount <= self._energy_reserves:
            self._energy_reserves -= amount
            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, MetabolicComponent,
                                        energy_reserves=self._energy_reserves)
            return True
        return False