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
from logging import Logger
from typing import Optional, Dict, List

from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.entities.flora import Flora
from biome.systems.managers.worldmap_manager import WorldMapManager
from shared.enums.enums import ComponentType, EntityType, TerrainType, DietType
from shared.enums.events import ComponentEvent
from shared.types import Position
from utils.loggers import LoggerManager
from shared.enums.strings import Loggers


class ForagingBehaviour:

    def __init__(self, worldmap_manager: WorldMapManager):
        self._logger: Logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._worldmap_manager: WorldMapManager = worldmap_manager
        self._target: Optional[Fauna] = None

    def set_target(self, target: Fauna) -> None:
        self._target = target

    def check_and_drink_water(self) -> bool:
        if not self._validate_target():
            return False

        position = self._target.get_position()
        if not position:
            return False

        try:
            terrain = self._worldmap_manager.get_terrain_at(position)

            if terrain in [TerrainType.WATER_SHALLOW]:
                current_thirst = self._target.thirst_level
                hydration_value = 5.0 + (100.0 - current_thirst) * 0.3
                self._target.consume_water(hydration_value)
                self._logger.debug(f"Consuming water: +{hydration_value} hydratation")
                return True

        except Exception as e:
            self._logger.warning(f"Error verifiying terrain while trying to drink: {e}")

        return False

    def check_and_eat_food(self) -> bool:
        if not self._validate_target():
            return False

        position = self._target.get_position()
        if not position:
            return False

        try:
            nearby_entities = self._worldmap_manager.get_entities_near(position, radius=1)
            diet_type = self._target.diet_type

            if diet_type == DietType.HERBIVORE:
                return self._process_herbivore_feeding(nearby_entities)
            elif diet_type == DietType.CARNIVORE:
                return self._process_carnivore_feeding(nearby_entities)
            elif diet_type == DietType.OMNIVORE:
                return self._process_omnivore_feeding(nearby_entities)

        except Exception as e:
            self._logger.warning(f"Error verifiying aliment: {e}")

        return False

    def _validate_target(self) -> bool:
        return self._target is not None and self._target.is_alive()

    def _process_herbivore_feeding(self, nearby_entities: Dict[EntityType, List[Entity]]) -> bool:
        flora_entities = nearby_entities.get(EntityType.FLORA, [])
        if not flora_entities:
            return False

        for flora in flora_entities:
            nutritive_value = flora.get_nutritive_value()
            flora.apply_damage(15.0, self._target.get_id())
            self._target.consume_vegetal(nutritive_value)
            self._logger.debug(f"Consuming plant: +{nutritive_value} nutrition")

            # TODO: Hacer esto bien, flora.die()  o algo asi y que tenga esto
            if random.random() < 0.1:
                flora.die()
                self._logger.debug(f"Plant {flora.get_id()} was completely consumed")
            return

    def _process_carnivore_feeding(self, nearby_entities: Dict[EntityType, List[Entity]]) -> bool:
        fauna_entities = nearby_entities.get(EntityType.FAUNA, [])
        if not fauna_entities:
            self._logger.debug(f"[HUNTING] No fauna entities nearby. Entities: {nearby_entities}")
            return False

        self._logger.debug(f"[HUNTING] Nearby fauna entities ({len(fauna_entities)} total):")
        for f in fauna_entities:
            info = (
                f"ID={f.get_id()}, SPECIES={f.get_species()}, "
                f"DIET={f.diet_type.name}, "
                f"VITALITY={f.vitality:.2f}/{f.max_vitality:.2f}, "
                f"ENERGY={f.energy_reserves:.2f}/{f.max_energy_reserves:.2f}, "
                f"HUNGER={f.hunger_level:.2f}, THIRST={f.thirst_level:.2f}, "
                f"STRESS={f.stress_level:.2f}"
            )
            self._logger.debug(f"[HUNTING] Entity -> {info}")

        potential_prey = []
        for prey in fauna_entities:
            if prey.diet_type != DietType.HERBIVORE:
                self._logger.debug(f"[REJECTED] ID={prey.get_id()} rejected: Not herbivore.")
                continue
            if prey.get_id() == self._target.get_id():
                self._logger.debug(f"[REJECTED] ID={prey.get_id()} rejected: Is self.")
                continue
            if prey.get_species() == self._target.get_species():
                self._logger.debug(f"[REJECTED] ID={prey.get_id()} rejected: Same species.")
                continue
            potential_prey.append(prey)

        if not potential_prey:
            self._logger.debug("[HUNTING] No valid prey found after filters.")
            return False

        prey = random.choice(potential_prey)
        self._logger.debug(f"[HUNTING] Prey selected: ID={prey.get_id()}, SPECIES={prey.get_species()}")

        nutritive_value = self._calculate_prey_nutritive_value(prey) + 4.0

        prey.apply_damage(15.0, self._target.get_id())
        self._target.consume_prey(nutritive_value)
        self._logger.debug(f"Consuming prey: +{nutritive_value} nutrition")

        hunt_success = random.random()
        if hunt_success < 0.2:
            vital_component = prey.get_component(ComponentType.VITAL)
            if vital_component:
                vital_component.vitality = 0
                vital_component.event_notifier.notify(
                    ComponentEvent.UPDATE_STATE,
                    ComponentType.VITAL,
                    vitality=vital_component.vitality
                )
                vital_component.event_notifier.notify(ComponentEvent.ENTITY_DEATH, ComponentType.VITAL)
                # TODO: NOTIFICAR ENTITY DEATH EN PREY
                self._logger.debug(f"Prey {prey.get_id()} was killed")

        return True

    def _process_omnivore_feeding(self, nearby_entities: Dict[EntityType, List[Entity]]) -> bool:
        # Primero que intente comer carne, que es más nutritiva
        if self._process_carnivore_feeding(nearby_entities):
            return True

        return self._process_herbivore_feeding(nearby_entities)

    def _calculate_prey_nutritive_value(self, prey: Fauna) -> float:
        nutritive_value = 1.0

        growth_component = prey.get_component(ComponentType.GROWTH)
        if growth_component:
            nutritive_value *= growth_component.current_size

        vital_component = prey.get_component(ComponentType.VITAL)
        if vital_component:
            nutritive_value *= (vital_component.vitality / vital_component.max_vitality)

        # Meto un factor adicional por ser carne
        nutritive_value *= 1.5

        return nutritive_value
