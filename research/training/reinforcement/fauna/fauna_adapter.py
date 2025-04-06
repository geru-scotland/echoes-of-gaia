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
from typing import Tuple, Optional, Dict

import numpy as np
from typing_extensions import Any

from biome.biome import Biome
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.managers.worldmap_manager import WorldMapManager
from config.settings import Settings
from research.training.reinforcement.adapter import EnvironmentAdapter
from research.training.reinforcement.config.training_config_manager import TrainingConfigManager
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import ComponentType, EntityType, SimulationMode, FaunaSpecies, Direction, FaunaAction, \
    PositionNotValidReason, TerrainType, BiomeType, DietType
from shared.enums.events import SimulationEvent
from shared.enums.strings import Loggers
from shared.types import Position, DecodedAction
from simulation.api.simulation_api import SimulationAPI
from simulation.core.systems.events.event_bus import SimulationEventBus
from utils.loggers import LoggerManager


class FaunaSimulationAdapter(EnvironmentAdapter):
    def __init__(self, fov_width: int, fov_height: int, fov_center):
        self._logger = LoggerManager.get_logger(Loggers.REINFORCEMENT)
        self._simulation_api: SimulationAPI = None
        self._biome: Biome = None
        self._worldmap_manager: WorldMapManager = None
        self._target: Fauna = None
        self._logger.info(f"Creating {self.__class__.__name__}...")
        self._fov_width: int = fov_width
        self._fov_height: int = fov_height
        self._fov_center: int = fov_center
        self._previous_position: Optional[Position] = None
        self._current_position: Optional[Position] = None
        self._visited_positions = set()
        self._heatmap = {}
        self._current_biome_type: Optional[BiomeType] = None
        self._previous_states: Dict[str, Any] = {}
        self._register_events()

    def __del__(self):
        try:
            self._logger.info("Closing Fauna Adapter resources...")
            self.finish_training_session()
        except Exception as e:
            self._logger.warning(f"Error during cleanup: {e}")

    def _register_events(self):
        SimulationEventBus.register(SimulationEvent.SIMULATION_TRAIN_TARGET_ACQUIRED, self._handle_target_acquired)

    def _handle_target_acquired(self, *args, **kwargs):
        entity: Entity = kwargs.get("entity", None)
        generation: int = kwargs.get("generation", None)

        if entity and generation and not TrainingTargetManager.is_acquired():
            self._target = entity
            TrainingTargetManager.mark_as_acquired()
            self._logger.info(
                f"Training target acquired: {entity.get_species()} (ID: {entity.get_id()}) - Generation {generation}")

    def _is_new_position(self, position: Position) -> bool:
        return position not in self._visited_positions

    def initialize(self):
        self._logger.info(f"Initializing {self.__class__.__name__} ...")
        random_config = TrainingConfigManager.generate_random_config("training.yaml")
        temp_config_path = TrainingConfigManager.save_temp_config(random_config)

        self._current_biome_type = BiomeType(random_config['biome']['type'])
        print(random_config)
        settings = Settings(override_configs=temp_config_path)
        SimulationEventBus.clear()
        BiomeEventBus.clear()

        self._register_events()
        self._simulation_api = SimulationAPI(
            settings=settings,
            mode=SimulationMode.TRAINING,
        )

        self._simulation_api.initialise_training()
        self._biome = self._simulation_api.get_biome()
        self._worldmap_manager = self._biome.get_worldmap_manager()

        TrainingTargetManager.reset()
        TrainingTargetManager.set_training_mode(True)

        target_species = self._select_target_species()
        # TODO: OJO! Que no se te olvide poner esto debidamente, la idea es
        # probar diversos rangos de generaciones.
        # TODO: Que coja 3-4 después de más runs quizá
        target_generation = random.randint(1, 4)

        TrainingTargetManager.set_target(
            str(target_species),
            EntityType.FAUNA,
            target_generation
        )

        self._logger.info(f"Waiting for target species: {target_species}, generation: {target_generation}")

        try:
            return self._wait_for_target_acquisition()
        except Exception as e:
            self._logger.warning(e)

    def _select_target_species(self) -> FaunaSpecies:
        available_species = list(FaunaSpecies)
        selected_species = random.choice(available_species)

        return FaunaSpecies.DEER

    def _wait_for_target_acquisition(self) -> bool:
        max_attempts = 5000
        attempts = 0

        self._logger.debug("Waiting for target acquisition...")

        while not TrainingTargetManager.is_acquired() and attempts < max_attempts:

            if not self._worldmap_manager.has_alive_entities(type=EntityType.FAUNA):
                self._logger.warning("There are no living entities in the worldmap, reseting.")
                break

            self._simulation_api.step(1)
            attempts += 1
            if attempts % 100 == 0:
                self._logger.debug(f"Waiting for target... {attempts} steps")

        if not TrainingTargetManager.is_acquired():
            self._logger.warning("Failed to acquire target after maximum attempts")
            return False

        self._logger.debug(f"Target acquired successfully: {TrainingTargetManager.get_current_episode()}")
        return True

    def step_environment(self, action: int, time_delta: int = 1) -> None:
        if not self._simulation_api or not self._target:
            return

        self._previous_position = self._target.get_position()

        nutrition_component = self._target.get_component(ComponentType.HETEROTROPHIC_NUTRITION)
        if nutrition_component and nutrition_component.energy_reserves <= 0:
            pass
        else:
            # Me puedo mover
            self._target.move(self._action_decode(action))

        self._current_position = self._target.get_position()

        self._check_and_drink_water()
        self._check_and_eat_food()

        self._simulation_api.step(time_delta)

    def _check_and_drink_water(self) -> None:
        if not self._target or not self._target.is_alive():
            return

        position = self._target.get_position()
        if not position:
            return

        try:
            terrain = self._worldmap_manager.get_terrain_at(position)

            if terrain in [TerrainType.WATER_SHALLOW]:
                # La cantidad de hidratación depende del nivel actual de sed
                # Cuanto más sediento, más hidratación, por ahora me cuadra
                current_thirst = self._target.thirst_level
                hydration_value = 5.0 + (100.0 - current_thirst) * 0.3
                self._target.consume_water(hydration_value)
                self._logger.debug(f"Consuming water: +{hydration_value} hydratation")
        except Exception as e:
            self._logger.warning(f"Error verifiying terrain while trying to drink: {e}")

    # TODO: Hacer un sistema de food y water, para gestionar esto
    def _check_and_eat_food(self) -> None:
        if not self._target or not self._target.is_alive():
            return

        position = self._target.get_position()
        if not position:
            return

        try:
            nearby_entities = self._worldmap_manager.get_entities_near(position, radius=1)

            diet_type = self._target.diet_type

            # Herbívoros
            if diet_type == DietType.HERBIVORE and nearby_entities[EntityType.FLORA]:
                for flora in nearby_entities[EntityType.FLORA]:
                    nutritive_value = flora.get_nutritive_value()
                    self._target.consume_vegetal(nutritive_value)
                    self._logger.info(f"Consuming plant: +{nutritive_value} nutrition")

                    if random.random() < 0.1:
                        self._worldmap_manager.remove_entity(flora.get_id())
                        self._logger.debug(f"Plant {flora.get_id()} was completely consumed")
                    return

            # Carnívoros
            elif diet_type == DietType.CARNIVORE and nearby_entities[EntityType.FAUNA]:
                potential_prey = [prey for prey in nearby_entities[EntityType.FAUNA]
                                  if hasattr(prey, 'diet_type') and prey.diet_type == DietType.HERBIVORE]

                if potential_prey:
                    prey = random.choice(potential_prey)

                    nutritive_value = 1.0
                    growth_component = prey.get_component(ComponentType.GROWTH)
                    if growth_component:
                        nutritive_value *= growth_component.current_size

                    vital_component = prey.get_component(ComponentType.VITAL)
                    if vital_component:
                        nutritive_value *= (vital_component.vitality / vital_component.max_vitality)

                    self._target.consume_prey(nutritive_value)
                    self._logger.debug(f"Consuming prey: +{nutritive_value} nutrition")

                    hunt_success = random.random()
                    if hunt_success < 0.2:
                        if vital_component:
                            vital_component.vitality = 0
                        self._logger.debug(f"Prey {prey.get_id()} was killed")
                    return

            elif diet_type == DietType.OMNIVORE:
                # Prioridad a proteína animal
                if nearby_entities[EntityType.FAUNA]:
                    potential_prey = [prey for prey in nearby_entities[EntityType.FAUNA]
                                      if hasattr(prey, 'diet_type') and prey.diet_type == DietType.HERBIVORE]

                    if potential_prey:
                        # TODO: Similar a la lógica de carnivoros, investigar un poco a ver diferencias
                        pass

                elif nearby_entities[EntityType.FLORA]:
                    # TODO: Lo mismo, echar un ojo a ver.
                    pass

        except Exception as e:
            self._logger.warning(f"Error checking for food: {e}")

    def handle_feeding(self, entity_id: int) -> bool:
        if entity_id not in self._worldmap_manager.g_entity_registry:
            return False

        entity = self._worldmap_manager.get_entity_by_id(entity_id)

        if not entity.is_alive() or entity.get_type() != EntityType.FAUNA:
            return False

        position = entity.get_position()
        if not position:
            return False

        nearby_entities = self._worldmap_manager.get_entities_near(position)

        # herviboros/omnivoros - los nutro con flora
        if entity.diet_type in [DietType.HERBIVORE, DietType.OMNIVORE] and nearby_entities[EntityType.FLORA]:
            for flora in nearby_entities[EntityType.FLORA]:
                nutritive_value = flora.get_nutritive_value()
                entity.consume_vegetal(nutritive_value)

                if random.random() < 0.1:  # TODO: Hacer esto bien, en función de la vitalidad de la flora etc.
                    self._worldmap_manager.remove_entity(flora.get_id())
                    self._logger.debug(f"Plant {flora.get_id()} was completely consumed and removed")

                return True

        # carnivoros/omnivoros - hago que se puedan nutrir de otra fauna
        if entity.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE] and nearby_entities[EntityType.FAUNA]:

            potential_prey = [prey for prey in nearby_entities[EntityType.FAUNA]
                              if prey.diet_type == DietType.HERBIVORE]

            if potential_prey:
                prey = random.choice(potential_prey)
                # Coger mejor de nutritive value
                nutritive_value = 1.0
                growth_component = prey.get_component(ComponentType.GROWTH)
                if growth_component:
                    nutritive_value *= growth_component.current_size

                vital_component = prey.get_component(ComponentType.VITAL)
                if vital_component:
                    nutritive_value *= (vital_component.vitality / vital_component.max_vitality)

                entity.consume_prey(nutritive_value)

                hunt_success = random.random()
                if hunt_success < 0.2:  # Lo mismo que con flora, hacer que vaya en función de ciertos parámetros.
                    prey.get_component(ComponentType.VITAL).vitality = 0
                    self._logger.debug(f"Prey {prey.get_id()} was killed and consumed")

                return True

        return False

    def compute_reward(self, action: FaunaAction):
        if not self._target or not self._target.is_alive():
            return -10.0

        reward: float = 0.0

        decoded_action: DecodedAction = self._action_decode(action)

        movement_reward: int = self.compute_movement_reward(decoded_action)

        physiological_reward: float = self.compute_physiological_state_reward()
        return reward + movement_reward + physiological_reward

    def _action_decode(self, action: int) -> DecodedAction:
        if action < len(Direction):
            return list(Direction)[action]

    def compute_physiological_state_reward(self) -> float:
        if not self._target or not self._target.is_alive():
            return 0.0

        reward: float = 0.0

        thirst_level: float = self._target.thirst_level / 100.0
        energy_ratio: float = self._target.energy_reserves / self._target.max_energy_reserves
        hunger_level: float = self._target.hunger_level / 100.0
        vitality: float = self._target.vitality / self._target.max_vitality
        stress_level: float = self._target.stress_level / 100.0

        thirst_ratio: float = 1.0 - thirst_level
        hunger_ratio: float = 1.0 - hunger_level

        reward += 0.05

        if energy_ratio > 0.6:
            reward += 0.1
        if thirst_ratio < 0.3:
            reward += 0.1
        if hunger_ratio < 0.3:
            reward += 0.1

        if vitality > 0.6:
            reward += 0.05
        if stress_level < 0.4:
            reward += 0.05

        if hunger_ratio > 0.85:
            reward -= 0.4
        if thirst_ratio > 0.85:
            reward -= 0.4
        if energy_ratio < 0.15:
            reward -= 0.3

        if vitality < 0.2:
            reward -= 0.1
        if stress_level > 0.8:
            reward -= 0.1

        if hasattr(self, '_previous_states'):
            prev_states = self._previous_states

            if 'thirst_ratio' in prev_states and thirst_ratio < prev_states['thirst_ratio']:
                improvement = prev_states['thirst_ratio'] - thirst_ratio
                reward += 0.8 * improvement

            if 'hunger_ratio' in prev_states and hunger_ratio < prev_states['hunger_ratio']:
                improvement = prev_states['hunger_ratio'] - hunger_ratio
                reward += 0.8 * improvement

            if 'energy_ratio' in prev_states and energy_ratio > prev_states['energy_ratio']:
                improvement = energy_ratio - prev_states['energy_ratio']
                reward += 0.6 * improvement

            if 'vitality' in prev_states:
                if vitality >= prev_states['vitality'] - 0.01:
                    reward += 0.2
                    self._logger.debug("Vitality maintained: +0.2")

            if 'stress_level' in prev_states:
                if stress_level <= prev_states['stress_level'] + 0.01:
                    reward += 0.2
                    self._logger.debug("Stress managed: +0.2")

        self._previous_states = {
            'thirst_ratio': thirst_ratio,
            'hunger_ratio': hunger_ratio,
            'energy_ratio': energy_ratio,
            'vitality': vitality,
            'stress_level': stress_level
        }

        return reward
    def compute_reward(self, action: FaunaAction):
        if not self._target or not self._target.is_alive():
            return -10.0

        reward: float = 0.0

        decoded_action: DecodedAction = self._action_decode(action)

        movement_reward: int = self.compute_movement_reward(decoded_action)
        water_reward: int = self.compute_water_reward()
        return reward + movement_reward + water_reward

    def _action_decode(self, action: int) -> DecodedAction:
        if action < len(Direction):
            return list(Direction)[action]

    def compute_movement_reward(self, direction: Direction) -> float:
        # self._logger.info(f"Computing movement reward for direction: {direction.name}")

        new_position: Position = self._target.movement_component.calculate_new_position(
            direction, self._previous_position
        )
        # self._logger.info(f"Calculated new position: {new_position} from previous: {self._previous_position}")

        is_valid, reason = self._worldmap_manager.is_valid_position(new_position, self._target.get_id())
        # self._logger.info(f"Position validation result: is_valid={is_valid}, reason={reason.name}")

        if not is_valid:
            if reason == PositionNotValidReason.POSITION_OUT_OF_BOUNDARIES:
                self._logger.debug("Movement rejected: position out of boundaries. Penalty: -1.0")
                return -1.0
            elif reason == PositionNotValidReason.POSITION_NON_TRAVERSABLE:
                self._logger.debug("Movement rejected: position non-traversable. Penalty: -0.8")
                return -0.8
            elif reason == PositionNotValidReason.POSITION_BUSY:
                self._logger.debug("Movement rejected: position busy. Penalty: -0.6")
                return -0.6

        reward = 0.0
        # self._logger.info("Movement accepted. Base reward: 0.0")

        if self._current_position == new_position and self._is_new_position(self._current_position):
            reward += 0.2
            self._visited_positions.add(self._current_position)
            self._logger.debug("Position is new. Exploration bonus applied: +0.4")

        self._logger.debug(f"Final reward returned: +{reward}")
        return reward

    def compute_water_reward(self) -> float:
        if not self._target or not self._target.is_alive():
            return 0.0

        reward = 0.0
        position = self._target.get_position()

        if not position:
            return 0.0

        try:
            terrain = self._worldmap_manager.get_terrain_at(position)

            thirst_ratio = 1.0 - (self._target.thirst_level / 100.0)

            if terrain in [TerrainType.WATER_SHALLOW]:
                # Aqui recompensa si la sed es alta, proporcionalemnte
                # TODO: Invertir numéricamente el thirsty level mejor
                if thirst_ratio > 0.7:  # MUY sediento
                    reward += 1
                    self._logger.debug("Drinking water because VERY thirsty: +1.2")
                elif thirst_ratio > 0.4:  # Moderadamente
                    reward += 0.4
                    self._logger.debug("Drinking water because moderately thirsty: +0.8")
                elif thirst_ratio > 0.1:  # Ligero
                    reward += 0.1
                    self._logger.debug("Drinking water while slightly thirsty: +0.3")

            if thirst_ratio > 0.9:
                reward -= 0.5
                self._logger.debug("Extremely thirsty penalization: -0.5")

            return reward

        except Exception as e:
            self._logger.warning(f"Errr computing water reard: {e}")
            return 0.0

    def get_observation(self):
        if not self._target:
            return self._get_default_observation()

        position = self._target.get_position()

        local_result = None
        if position:
            local_result = self._worldmap_manager.get_local_maps(position, self._fov_width, self._fov_height)

        if local_result is None:
            local_fov_terrain = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
            validity_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
            visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
            flora_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
            fauna_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        else:
            local_fov_terrain, validity_mask, flora_mask, fauna_mask = local_result

            local_fov_terrain = local_fov_terrain.astype(np.int64)
            visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)

            center_y, center_x = self._fov_height // 2, self._fov_width // 2

            for y in range(self._fov_height):
                for x in range(self._fov_width):
                    # Convierto coords de esta celda, a posición global
                    global_y = position[0] + (y - center_y)
                    global_x = position[1] + (x - center_x)
                    global_pos = (global_y, global_x)

                    if global_pos in self._visited_positions:
                        visited_mask[y, x] = 1.0

        validity_map = validity_mask.astype(np.float32)
        visited_map = visited_mask.astype(np.float32)
        flora_map = flora_mask.astype(np.float32)
        fauna_map = fauna_mask.astype(np.float32)

        thirst_level = 0.0
        energy_reserves = 0.0
        biome_type_idx = list(BiomeType).index(self._biome.get_biome_type())

        vitality = 0.0
        stress_level = 0.0
        hunger_level = 0.

        if self._target and self._target.is_alive():
            thirst_level = self._target.thirst_level / 100.0  # OJO, normalizo, que si no no tiraba bien.
            energy_reserves = self._target.energy_reserves / self._target.max_energy_reserves

            vitality = self._target.vitality / self._target.max_vitality
            stress_level = self._target.stress_level / 100.0
            hunger_level = self._target.hunger_level / 100.0

        return {
            "biome_type": biome_type_idx,
            "terrain_map": local_fov_terrain,
            "validity_map": validity_map,
            "visited_map": visited_map,
            "flora_map": flora_map,
            "fauna_map": fauna_map,
            "thirst_level": np.array([thirst_level], dtype=np.float32),
            "energy_reserves": np.array([energy_reserves], dtype=np.float32),
            "vitality": np.array([vitality], dtype=np.float32),
            "stress_level": np.array([stress_level], dtype=np.float32),
            "hunger_level": np.array([hunger_level], dtype=np.float32)
        }

    def _find_nearby_entities(self, position, radius):
        return []

    def _get_default_observation(self):
        terrain_map = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
        valid_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)

        return {
            "biome_type": BiomeType.TROPICAL,
            "terrain_map": terrain_map,
            "validity_map": valid_mask,
            "visited_map": visited_mask,
            "thirst_level": np.array([1.0], dtype=np.float32),  # Por defecto, sin sed (invertido, tengo que cambiar)
            "energy_reserves": np.array([1.0], dtype=np.float32),
            "vitality": np.array([1.0], dtype=np.float32),
            "stress_level": np.array([0.0], dtype=np.float32),
            "hunger_level": np.array([0.0], dtype=np.float32),
        }

    def finish_training_session(self):
        if self._simulation_api:
            self._simulation_api.finish_training()

        TrainingTargetManager.set_training_mode(False)

    def get_target(self) -> Entity:
        return self._target
