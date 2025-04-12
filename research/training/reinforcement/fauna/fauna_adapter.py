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
from typing import Tuple, Optional, Dict, Set

import numpy as np
from typing_extensions import Any, List

from biome.biome import Biome
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.systems.behaviours.foraging import ForagingBehaviour
from biome.systems.behaviours.pursuit_and_flee import PursuitAndFleeBehaviour
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.managers.worldmap_manager import WorldMapManager
from config.settings import Settings
from research.training.reinforcement.adapter import EnvironmentAdapter
from research.training.reinforcement.config.training_config_manager import TrainingConfigManager
from research.training.reinforcement.fauna.entity_interaction import LocalInteractionSimulator
from research.training.reinforcement.fauna.training_target_manager import TrainingTargetManager
from shared.enums.enums import ComponentType, EntityType, SimulationMode, FaunaSpecies, Direction, FaunaAction, \
    PositionNotValidReason, TerrainType, BiomeType, DietType
from shared.enums.events import SimulationEvent
from shared.enums.strings import Loggers
from shared.timers import Timers
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
        self._visited_positions: Set = set()
        self._heatmap = {}
        self._current_biome_type: Optional[BiomeType] = None
        self._previous_states: Dict[str, Any] = {}
        self._foraging_behaviour: Optional[ForagingBehaviour] = None
        # self._pursuit_behaviour: Optional[PursuitBehaviour] = None
        # self._entity_interaction_system: Optional[EntityInteractionAI] = None
        self._entity_behaviour_system: Optional[PursuitAndFleeBehaviour] = None
        self._local_interaction_simulator: Optional[LocalInteractionSimulator] = None
        self._available_fauna: Optional[List[FaunaSpecies]] = None
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

            # if self._foraging_behaviour and self._pursuit_behaviour:
            if self._foraging_behaviour:
                self._foraging_behaviour.set_target(self._target)
                self._entity_behaviour_system.set_target(self._target)
                self._local_interaction_simulator.set_target(self._target)
                # self._pursuit_behaviour.set_target(self._target)
                TrainingTargetManager.mark_as_acquired(self._target.get_id())
                self._logger.info(
                    f"Training target acquired: {entity.get_species()} (ID: {entity.get_id()}) - Generation {generation}")

    def _is_new_position(self, position: Position) -> bool:
        return position not in self._visited_positions

    def initialize(self):
        self._logger.info(f"Initializing {self.__class__.__name__} ...")
        random_config = TrainingConfigManager.generate_random_config("training.yaml")
        temp_config_path = TrainingConfigManager.save_temp_config(random_config)

        self._current_biome_type = BiomeType(random_config['biome']['type'])
        self._available_fauna = random_config['biome']['available_fauna']

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

        self._foraging_behaviour = ForagingBehaviour(self._worldmap_manager)
        # self._pursuit_behaviour = PursuitBehaviour(self._worldmap_manager)
        # self._entity_interaction_system = EntityInteractionAI(self._pursuit_behaviour)
        self._entity_behaviour_system = PursuitAndFleeBehaviour(self._worldmap_manager)
        self._local_interaction_simulator = LocalInteractionSimulator(self._worldmap_manager)
        TrainingTargetManager.reset()
        TrainingTargetManager.set_training_mode(SimulationMode.TRAINING)

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
        selected_species = random.choice(self._available_fauna)
        return selected_species

    def _wait_for_target_acquisition(self) -> bool:
        max_attempts = 5000
        attempts = 0

        self._logger.debug("Waiting for target acquisition...")

        while not TrainingTargetManager.is_acquired() and attempts < max_attempts:

            if not self._worldmap_manager.has_alive_entities(type=EntityType.FAUNA,
                                                             species=TrainingTargetManager.get_target()[0]):
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

        if self._target.energy_reserves > 0:
            # Me puedo mover
            direction: Direction = self._action_decode(action)

            if direction:
                self._target.move(direction)

            self._current_position = self._target.get_position()

            if self._current_position != self._previous_position and self._foraging_behaviour:
                self._foraging_behaviour.check_and_drink_water()
                self._foraging_behaviour.check_and_eat_food()

        if self._current_position:
            radius = 5
            nearby_entities = self._worldmap_manager.get_entities_near(self._current_position, radius=radius)

            fauna_entities = nearby_entities.get(EntityType.FAUNA, [])

            self._entity_behaviour_system.process_entity_behaviours(fauna_entities)

            self._local_interaction_simulator.process_interactions()

        self._simulation_api.step(time_delta)

    def compute_reward(self, action: FaunaAction):
        if not self._target or not self._target.is_alive():
            time_lived_ratio: float = 1 - (
                    self._target.get_age_after_death() / (self._target.lifespan * Timers.Calendar.REAL_YEAR))
            penalization: float = -15 * time_lived_ratio
            self._logger.info(self._target.get_age_after_death())
            self._logger.info(self._target.lifespan * Timers.Calendar.REAL_YEAR)
            self._logger.info(f"DIED WITH RATIO: {time_lived_ratio}, granted {penalization} points penalization")
            return penalization

        reward: float = 0.0
        movement_reward: float = 0.0
        pursuit_behaviour_reward: float = 0.0

        decoded_action: DecodedAction = self._action_decode(action)

        if decoded_action:
            movement_reward: int = self.compute_movement_reward(decoded_action)

        physiological_reward: float = self.compute_physiological_state_reward()

        if decoded_action:
            pursuit_behaviour_reward = self.compute_pursuit_reward(decoded_action)

        return reward + movement_reward + physiological_reward + pursuit_behaviour_reward

    def _action_decode(self, action: int) -> DecodedAction:
        if action < len(Direction):
            return list(Direction)[action]

    def compute_physiological_state_reward(self) -> float:
        if not self._target or not self._target.is_alive():
            return 0.0

        reward: float = 0.0

        thirst_ratio: float = self._target.thirst_level / 100.0
        energy_ratio: float = self._target.energy_reserves / self._target.max_energy_reserves
        hunger_ratio: float = self._target.hunger_level / 100.0
        vitality: float = self._target.vitality / self._target.max_vitality
        stress_level: float = self._target.stress_level / 100.0

        somatic_integrity: float = self._target.somatic_integrity / self._target.max_somatic_integrity

        reward += 0.02

        if energy_ratio > 0.5:
            reward += 0.3
        if thirst_ratio > 0.4:
            reward += 0.5
        if hunger_ratio > 0.3:
            reward += 0.5
        if somatic_integrity > 0.5:
            reward += 0.3

        if vitality > 0.6:
            reward += 0.05
        if stress_level < 0.4:
            reward += 0.05

        if hunger_ratio < 0.2 or thirst_ratio < 0.2:
            reward -= 0.8

        if energy_ratio < 0.15:
            reward -= 0.3

        if vitality < 0.2:
            reward -= 0.1

        if stress_level > 0.8:
            reward -= 0.2

        if hasattr(self, '_previous_states'):
            prev_states = self._previous_states

            if 'thirst_ratio' in prev_states and thirst_ratio > prev_states['thirst_ratio']:
                improvement = thirst_ratio - prev_states['thirst_ratio']
                reward += 3 * improvement

            if 'hunger_ratio' in prev_states and hunger_ratio > prev_states['hunger_ratio']:
                improvement = hunger_ratio - prev_states['hunger_ratio']
                reward += 3 * improvement

            if 'energy_ratio' in prev_states and energy_ratio > prev_states['energy_ratio']:
                improvement = energy_ratio - prev_states['energy_ratio']
                reward += 2 * improvement

            # if 'vitality' in prev_states:
            #     if vitality >= prev_states['vitality'] - 0.01:
            #         reward += 0.2
            #         self._logger.debug("Vitality maintained: +0.2")

            if 'stress_level' in prev_states:
                if stress_level <= 0.3:
                    reward += 0.1
                    self._logger.debug("Stress managed: +0.2")

            if 'somatic_integrity' in prev_states:
                if somatic_integrity < prev_states['somatic_integrity']:
                    reward -= 0.3
                    self._logger.debug("Stress managed: +0.2")

        self._previous_states = {
            'thirst_ratio': thirst_ratio,
            'hunger_ratio': hunger_ratio,
            'energy_ratio': energy_ratio,
            'vitality': vitality,
            'stress_level': stress_level,
            'somatic_integrity': somatic_integrity
        }

        return reward

    def compute_movement_reward(self, direction: Direction) -> float:
        # self._logger.info(f"Computing movement reward for direction: {direction.name}")

        new_position: Position = self._target.movement_component.calculate_new_position(
            direction, self._previous_position
        )
        # self._logger.info(f"Calculated new position: {new_position} from previous: {self._previous_position}")

        is_valid, reason = self._worldmap_manager.is_valid_position(new_position, self._target.get_id())
        # self._logger.info(f"Position validation result: is_valid={is_valid}, reason={reason.name}")

        if not is_valid:
            return -1.0

        reward = 0.0
        # self._logger.info("Movement accepted. Base reward: 0.0")

        if self._current_position == new_position and self._is_new_position(self._current_position):
            if self._target.thirst_level > 0.2 and self._target.hunger_level > 0.2:
                reward += 0.4

            self._visited_positions.add(self._current_position)
            self._logger.debug("Position is new. Exploration bonus applied: +0.4")

        self._logger.debug(f"Final reward returned: +{reward}")
        return reward

    def compute_pursuit_reward(self, direction: Direction) -> float:
        if not self._target or not self._target.is_alive():
            return 0.0

        reward = 0.0
        position = self._target.get_position()
        if not position:
            return 0.0

        nearby_entities = self._worldmap_manager.get_entities_near(position, radius=3)

        predators, potential_prey = [], []

        fauna_entities = nearby_entities.get(EntityType.FAUNA, [])
        for entity in fauna_entities:
            if not isinstance(entity, Fauna) or not entity.is_alive() or entity.get_id() == self._target.get_id() \
                    or entity.get_species() == self._target.get_species():
                continue

            if (entity.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE] and
                    self._target.diet_type == DietType.HERBIVORE):
                predators.append(entity)

            if (self._target.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE] and
                    entity.diet_type == DietType.HERBIVORE):
                potential_prey.append(entity)

        if self._target.diet_type == DietType.HERBIVORE and predators:
            my_pos = self._target.get_position()
            for predator in predators:
                other_pos = predator.get_position()
                if not other_pos:
                    continue

                moving_away = self._local_interaction_simulator.is_moving_away_from(my_pos, other_pos, direction)
                distance = self._local_interaction_simulator.calculate_manhattan_distance(my_pos, other_pos)

                threat_factor = max(1.0, min(1.0, 3.0 / distance))

                if moving_away:
                    reward += 0.6 * threat_factor
                    self._logger.debug(f"Prey fleeing from predator: +{0.5 * threat_factor:.3f}")
                else:
                    reward -= 0.8 * threat_factor
                    self._logger.debug(f"Prey moving toward predator: -{0.8 * threat_factor:.3f}")

        elif self._target.diet_type in [DietType.CARNIVORE, DietType.OMNIVORE] and potential_prey:
            my_pos = self._target.get_position()
            for prey in potential_prey:
                other_pos = prey.get_position()
                if not other_pos:
                    continue

                hunger_factor = (100.0 - self._target.hunger_level) / 100.0
                energy_factor = self._target.energy_reserves / self._target.max_energy_reserves
                hunting_motivation = hunger_factor * energy_factor

                distance = self._local_interaction_simulator.calculate_manhattan_distance(my_pos, other_pos)
                proximity_bonus = max(0.1, min(1.0, 3.0 / distance))

                moving_toward = self._local_interaction_simulator.is_moving_toward(my_pos, other_pos, direction)

                if moving_toward:
                    pursuit_reward = 0.6 * hunting_motivation * proximity_bonus
                    reward += pursuit_reward
                    self._logger.debug(f"Predator pursuing prey: +{pursuit_reward:.3f}")
                else:
                    if hunger_factor > 0.6:
                        avoidance_penalty = 0.1 * hunting_motivation * proximity_bonus
                        reward -= avoidance_penalty
                        self._logger.debug(f"Predator avoiding prey while hungry: -{avoidance_penalty:.3f}")
        return reward

    def get_observation(self):
        if not self._target:
            return self._get_default_observation()

        position = self._target.get_position()

        local_result = None
        if position:
            local_result = self._worldmap_manager.get_local_maps(position, self._target.diet_type,
                                                                 self._target.get_species(), self._fov_width,
                                                                 self._fov_height)

        if local_result is None:
            local_fov_terrain = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
            validity_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
            visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.bool_)
            flora_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
            prey_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
            predator_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
            water_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
            food_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        else:
            local_fov_terrain, validity_mask, flora_mask, prey_mask, predator_mask, water_mask, food_mask = local_result

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
        prey_map = prey_mask.astype(np.float32)
        predator_map = predator_mask.astype(np.float32)
        water_map = water_mask.astype(np.float32)
        food_map = food_mask.astype(np.float32)

        thirst_level = 0.0
        energy_reserves = 0.0
        biome_type_idx = list(BiomeType).index(self._biome.get_biome_type())
        diet_type_idx = list(DietType).index(self._target.diet_type)

        vitality = 0.0
        stress_level = 0.0
        hunger_level = 0.0
        somatic_integrity = 1.0

        if self._target and self._target.is_alive():
            thirst_level = self._target.thirst_level / 100.0  # OJO, normalizo, que si no no tiraba bien.
            energy_reserves = self._target.energy_reserves / self._target.max_energy_reserves

            vitality = self._target.vitality / self._target.max_vitality
            stress_level = self._target.stress_level / 100.0
            hunger_level = self._target.hunger_level / 100.0
            somatic_integrity = self._target.somatic_integrity / self._target.max_somatic_integrity

        return {
            "biome_type": biome_type_idx,
            "diet_type": diet_type_idx,
            "terrain_map": local_fov_terrain,
            "validity_map": validity_map,
            "visited_map": visited_map,
            "flora_map": flora_map,
            "prey_map": prey_map,
            "predator_map": predator_map,
            "water_map": water_map,
            "food_map": food_map,
            "thirst_level": np.array([thirst_level], dtype=np.float32),
            "energy_reserves": np.array([energy_reserves], dtype=np.float32),
            "vitality": np.array([vitality], dtype=np.float32),
            "stress_level": np.array([stress_level], dtype=np.float32),
            "hunger_level": np.array([hunger_level], dtype=np.float32),
            "somatic_integrity": np.array([somatic_integrity], dtype=np.float32),
        }

    def _get_default_observation(self):
        terrain_map = np.full((self._fov_height, self._fov_width), TerrainType.UNKNWON.value, dtype=np.int64)
        valid_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        visited_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.float32)
        prey_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        predator_mask = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        water_map = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        food_map = np.zeros((self._fov_height, self._fov_width), dtype=np.int8)
        return {
            "biome_type": list(BiomeType).index(BiomeType.TROPICAL),
            "diet_type": list(DietType).index(DietType.HERBIVORE),
            "terrain_map": terrain_map,
            "validity_map": valid_mask,
            "visited_map": visited_mask,
            "prey_map": prey_mask,
            "predator_map": predator_mask,
            "water_map": water_map,
            "food_map": food_map,
            "thirst_level": np.array([1.0], dtype=np.float32),  # Por defecto, sin sed (invertido, tengo que cambiar)
            "energy_reserves": np.array([1.0], dtype=np.float32),
            "vitality": np.array([1.0], dtype=np.float32),
            "stress_level": np.array([0.0], dtype=np.float32),
            "hunger_level": np.array([0.0], dtype=np.float32),
            "somatic_integrity": np.array([1.0], dtype=np.float32),
        }

    def finish_training_session(self):
        if self._simulation_api:
            self._simulation_api.finish_training()

        TrainingTargetManager.set_training_mode(SimulationMode.TRAINING_FINISHED)

    def get_target(self) -> Entity:
        return self._target
