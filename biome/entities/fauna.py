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
from typing import Any, Set, Optional

from biome.components.physiological.heterotrophic_nutrition import HeterotrophicNutritionComponent
from biome.components.physiological.vital import VitalComponent
from biome.entities.descriptor import EntityDescriptor
from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums.enums import FaunaSpecies, ComponentType, DietType, Direction
from shared.types import HabitatList, Position


class Fauna(Entity):

    def __init__(self, id: int, env: simpyEnv, fauna_type: FaunaSpecies, habitats: HabitatList, lifespan: float,
                 evolution_cycle: int = 0, diet_type: DietType = DietType.HERBIVORE):
        descriptor: EntityDescriptor = EntityDescriptor.create_fauna(fauna_type)
        super().__init__(id, env, descriptor, habitats, lifespan, evolution_cycle)
        self._fauna_type = fauna_type
        self._diet_type = diet_type
        self._logger.debug(f"FAUNA CREATED: {fauna_type}")
        self._habitats: HabitatList = habitats
        self._visited_positions: Set = set()

    def _register_events(self):
        super()._register_events()

    def dump_components(self) -> None:
        pass

    def compute_state(self):
        pass

    def consume_water(self, hidration_value: float):
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            nutrition_component.consume_water(hidration_value)

    def consume_vegetal(self, nutritive_value: float):
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            nutrition_component.consume_food(nutritive_value)
            self._logger.debug(f"Consuming plant: +{nutritive_value} nutrition")

    def consume_prey(self, nutritive_value: float):
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            enhanced_value = nutritive_value * 1.5
            nutrition_component.consume_food(enhanced_value)
            self._logger.debug(f"Consuming prey: +{enhanced_value} nutrition")

    def move(self, direction: Direction):
        super().move(direction)
        new_position: Position = self.get_position()

        if new_position not in self._visited_positions:
            self._visited_positions.add(new_position)

    def apply_damage(self, damage_amount: float, source_entity_id: Optional[int] = None) -> None:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            vital_component.apply_damage(damage_amount, source_entity_id)

    def heal_integrity(self, healing_amount: float) -> None:
        vital_component = self.get_component(ComponentType.VITAL)
        if vital_component:
            vital_component.heal_integrity(healing_amount)

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

    @property
    def thirst_level(self) -> float:
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            return nutrition_component.thirst_level

    @property
    def hunger_level(self) -> float:
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            return nutrition_component.hunger_level

    @property
    def diet_type(self) -> DietType:
        return self._diet_type

    @property
    def energy_reserves(self) -> float:
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            return nutrition_component.energy_reserves

    @property
    def max_energy_reserves(self) -> float:
        nutrition_component: HeterotrophicNutritionComponent = self._components.get(
            ComponentType.HETEROTROPHIC_NUTRITION, None)
        if self.components and nutrition_component:
            return nutrition_component.max_energy_reserves

    @property
    def vitality(self) -> float:
        vital_component: VitalComponent = self._components.get(ComponentType.VITAL, None)
        if self.components and vital_component:
            return vital_component.vitality
        return 0.0

    @property
    def max_vitality(self) -> float:
        vital_component: VitalComponent = self._components.get(ComponentType.VITAL, None)
        if self.components and vital_component:
            return vital_component.max_vitality
        return 0.0

    @property
    def stress_level(self) -> float:
        vital_component: VitalComponent = self._components.get(ComponentType.VITAL, None)
        if self.components and vital_component:
            return vital_component.stress_handler.stress_level
        return 0.0

    @property
    def visited_positions(self) -> Set:
        return self._visited_positions

# TODO: Getters para stres
