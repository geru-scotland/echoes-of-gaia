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
from typing import Optional, Dict, Any

from biome.components.base.component import EntityComponent
from simpy import Environment as simpyEnv

from biome.components.handlers.energy_handler import EnergyHandler
from biome.components.handlers.stress_handler import StressHandler
from biome.systems.components.registry import ComponentRegistry
from biome.systems.events.event_notifier import EventNotifier
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.enums.thresholds import MetabolicThresholds
from shared.timers import Timers


class PhotosyntheticMetabolismComponent(EntityComponent):
    def __init__(self, env: simpyEnv, event_notifier: EventNotifier, lifespan: float,
                 photosynthesis_efficiency: float = 0.75, respiration_rate: float = 0.05,
                 metabolic_activity: float = 1.0, max_energy_reserves: float = 100.0,
                 optimal_respiration_ratio: float = 0.2):

        self._stress_handler: StressHandler = StressHandler(event_notifier, lifespan)
        self._energy_handler: EnergyHandler = EnergyHandler(event_notifier, max_energy_reserves)

        super().__init__(env, ComponentType.PHOTOSYNTHETIC_METABOLISM, event_notifier, lifespan)

        self._base_photosynthesis_efficiency: float = round(photosynthesis_efficiency, 2)
        self._photosynthesis_efficiency: float = self._base_photosynthesis_efficiency
        self._base_respiration_rate: float = round(respiration_rate, 4)
        self._respiration_rate: float = self._base_respiration_rate
        self._metabolic_activity: float = round(metabolic_activity, 2)
        # TODO: Para light availability, hace falta más información en el state
        # quizá agregar el weather effect y analizar desde aquí, mediante ClimateService
        self._light_availability: float = 0.7
        self._temperature_modifier: float = 1.0
        self._water_modifier: float = 0.7
        self._biological_age: Optional[float] = None
        self._optimal_respiration_ratio: float = optimal_respiration_ratio
        self._energy_factor: float = self._energy_handler.energy_reserves / self._energy_handler.max_energy_reserves


        self._logger.debug(f"Metabolic component initialized: Photosynthesis={self._photosynthesis_efficiency}, "
                           f"Respiration={self._respiration_rate}, Energy={self._energy_handler.energy_reserves}")

        ComponentRegistry.get_photosynthetic_metabolism_manager().register_component(id(self), self)

    def _register_events(self):
        super()._register_events()
        self._stress_handler.register_events()
        self._energy_handler.register_events()
        self._event_notifier.register(ComponentEvent.ENERGY_UPDATED, self._handle_energy_update)
        self._event_notifier.register(ComponentEvent.BIOLOGICAL_AGE_UPDATED, self._handle_biological_age_update)

    def _handle_biological_age_update(self, *args, **kwargs) -> None:
        self._biological_age = kwargs.get("biological_age", None)

    def _handle_energy_update(self, *args, **kwargs) -> None:
        energy_reserves: float = kwargs.get("energy_reserves", 0.0)
        self._energy_factor = energy_reserves / self._energy_handler.max_energy_reserves

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
        if amount <= self._energy_handler.energy_reserves:
            new_energy: float = self._energy_handler.energy_reserves - amount
            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, PhotosyntheticMetabolismComponent,
                                        energy_reserves=new_energy)
            return True
        return False

    def get_state(self) -> Dict[str, Any]:
        return {
            "energy_reserves": self._energy_handler.energy_reserves,
            "max_energy_reserves": self._energy_handler.max_energy_reserves,
            "photosynthesis_efficiency": self._photosynthesis_efficiency,
            "respiration_rate": self._respiration_rate,
            "metabolic_activity": self._metabolic_activity,
        }

    @property
    def base_photosynthesis_efficiency(self) -> float:
        return self._base_photosynthesis_efficiency

    @property
    def base_respiration_rate(self) -> float:
        return self._base_respiration_rate

    @property
    def metabolic_activity(self) -> float:
        return self._metabolic_activity

    @property
    def energy_reserves(self) -> float:
        return self._energy_handler.energy_reserves

    @property
    def max_energy_reserves(self) -> float:
        return self._energy_handler.max_energy_reserves