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

"""
Flora entity implementation with dormancy and photosynthetic systems.

Extends Entity for plant behavior with dormancy state management;
handles environmental stress and weather adaptation mechanisms.
Provides nutritive value calculation and integrity tracking - supports
damage systems and photosynthetic response to climate events.
"""

from typing import Set, Type, Dict, Optional

from biome.entities.descriptor import EntityDescriptor
from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums.enums import FloraSpecies, ComponentType, WeatherEvent
from shared.enums.events import ComponentEvent
from shared.enums.reasons import DormancyReason, StressReason
from shared.types import HabitatList


class Flora(Entity):
    def __init__(self, id: int, env: simpyEnv, flora_type: FloraSpecies, habitats: HabitatList, lifespan: float,
                 evolution_cycle: int = 0):
        descriptor: EntityDescriptor = EntityDescriptor.create_flora(flora_type)
        super().__init__(id, env, descriptor, habitats, lifespan, evolution_cycle)
        self._logger.debug(f"Flora entity initialized: {flora_type}")
        self._component_dormancy_reasons: Dict[Type, Set[DormancyReason]] = {}
        self._flora_type: FloraSpecies = flora_type
        self._state.update("is_dormant", False)

    def _register_events(self):
        super()._register_events()
        self._event_notifier.register(ComponentEvent.DORMANCY_REASONS_CHANGED, self._handle_dormancy_reasons_changed)

    def _handle_dormancy_reasons_changed(self, component: Type, reasons: Set[DormancyReason]) -> None:
        self._logger.debug(f"Component: {component} Reason: {reasons}")
        self._component_dormancy_reasons[component] = reasons

        self._update_dormancy_state()

    def _update_dormancy_state(self) -> None:
        self._logger.warning(self._state)
        any_dormancy_reason = any(
            len(reasons) > 0
            for reasons in self._component_dormancy_reasons.values()
        )

        if any_dormancy_reason != self._state.get("is_dormant"):
            self._state.update("is_dormant", any_dormancy_reason)

            self._event_notifier.notify(ComponentEvent.DORMANCY_UPDATED,
                                        dormant=any_dormancy_reason)

    def _handle_toggle_dormancy(self, *args, **kwargs) -> None:
        dormant: bool = kwargs.get("dormant", False)
        self._event_notifier.notify(ComponentEvent.DORMANCY_UPDATED, dormant=dormant)

    def compute_state(self):
        pass

    def get_nutritive_value(self) -> float:
        autotrophic_component = self.get_component(ComponentType.AUTOTROPHIC_NUTRITION)
        if autotrophic_component:
            return autotrophic_component.current_nutritive_value, autotrophic_component.current_toxicity
        return 0.5, 0.1

    def dump_components(self) -> None:
        if not self._components:
            self._logger.error(f"Entity '{self._flora_type}' has no components.")
            return

        self._logger.debug(f"Entity '{self._flora_type}' Components:")
        for component_type, component in self._components.items():
            component_attrs = vars(component)
            formatted_attrs = ", ".join(
                f"{k}={v.__class__}" for k, v in component_attrs.items() if not k.startswith("_"))
            self._logger.debug(f" - {component_type}: {formatted_attrs}")

    def apply_damage(self, damage_amount: float, source_entity_id: Optional[int] = None) -> None:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            vital_component.apply_damage(damage_amount, source_entity_id)

    def heal_integrity(self, healing_amount: float) -> None:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            vital_component.heal_integrity(healing_amount)

    def handle_weather_event(self, weather_event: WeatherEvent, temperature: float) -> None:
        if not self.is_alive():
            return

        photo_component = self.get_component(ComponentType.PHOTOSYNTHETIC_METABOLISM)
        vital_component = self.get_component(ComponentType.VITAL)

        if weather_event == WeatherEvent.RAIN or weather_event == WeatherEvent.THUNDERSTORM:
            if photo_component:
                photo_component.water_modifier = min(1.0, photo_component.water_modifier + 0.2)
                self._logger.debug(f"Rain increased water modifier to {photo_component.water_modifier:.2f}")

        elif weather_event == WeatherEvent.DROUGHT:
            if photo_component:
                photo_component.water_modifier = max(0.1, photo_component.water_modifier - 0.15)
                self._logger.debug(f"Drought decreased water modifier to {photo_component.water_modifier:.2f}")

        if temperature > 35.0 and vital_component:
            vital_component.stress_handler.modify_stress(0.1, StressReason.TEMPERATURE_EXTREME)
            self._logger.debug(f"High temperature caused stress increase")

        elif temperature < 0.0 and vital_component:
            vital_component.stress_handler.modify_stress(0.15, StressReason.TEMPERATURE_EXTREME)
            self._logger.debug(f"Low temperature caused stress increase")

    @property
    def somatic_integrity(self) -> float:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            return vital_component.somatic_integrity
        return 0.0

    @property
    def max_somatic_integrity(self) -> float:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            return vital_component.max_somatic_integrity
        return 0.0

    @property
    def integrity_percentage(self) -> float:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            return vital_component.get_integrity_percentage()
        return 0.0
