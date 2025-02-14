import logging
from logging import Logger
from typing import List, Dict, Any

from simpy import Environment as simpyEnv

from biome.components.registry import get_component_class
from biome.entities.entity import Entity
from biome.entities.fauna import Fauna
from biome.entities.flora import Flora
from shared.enums import FloraType, FaunaType
from shared.stores.biome_store import BiomeStore
from shared.strings import Loggers
from shared.types import TileMap, Spawns, EntityList
from utils.loggers import setup_logger


class WorldMapManager:
    class SpawnSystem:
        def __init__(self, env: simpyEnv, flora_spawns: Spawns = None, fauna_spawns: Spawns = None):
            self._logger: Logger = setup_logger("spawn_system", "spawns.log")
            self._env: simpyEnv = env
            self._spawned_flora: EntityList = self._spawn_entities(flora_spawns, Flora, FloraType, BiomeStore.flora)
            self._spawned_fauna: EntityList = self._spawn_entities(fauna_spawns, Fauna, FaunaType, BiomeStore.fauna)


        def _spawn_entities(self, spawns: Spawns, entity_class, entity_type_enum, biome_store) -> EntityList:
            if not spawns:
                return []

            self._logger.info(f"Spawning {entity_class.__name__}...")

            spawned_entities = []

            for spawn in spawns:
                try:
                    entity_type = entity_type_enum(str(spawn.get("type")).lower())
                    amount = spawn.get("spawns")

                    if amount < 1 or amount > 30:
                        raise ValueError(f"Invalid spawn amount: {amount}. Must be between 1 and 30.")

                except (AttributeError, ValueError) as e:
                    self._logger.exception(f"There was an error loading {entity_class.__name__} spawns: {e}")
                    continue

                components = spawn.get("components")

                if not components:
                    store_components: List[str] = biome_store.get(spawn.get("type"), None).get("components", [])
                    if not store_components:
                        self._logger.warning(f"Spawn doesn't have components: {spawn}")
                        spawned_entities.extend([entity_class(self._env, entity_type) for _ in range(amount)])
                        continue
                    components = [
                        {cmp: BiomeStore.components.get(cmp, {}).get("defaults", {})}
                        for cmp in store_components
                        if BiomeStore.components.get(cmp)
                    ]

                for _ in range(amount):
                    entity: Entity = entity_class(self._env, entity_type)

                    for component in components:
                        for class_name, attribute_dict in component.items():
                            defaults = BiomeStore.components.get(class_name, {}).get("defaults", {}).copy()
                            # Primera vez con esta sintaxis
                            # desempaqueto defaults, y luego, los atributos del value del dict
                            # esos atributos, van a sobreescribir, asi que es un merge
                            # lo que no esté en uno, está en el otro, y los valores finales son los de attrib.
                            data = {**defaults, **(attribute_dict or {})}

                            if data:
                                component_class = get_component_class(class_name)
                                if component_class:
                                    component_instance = component_class(self._env, **data)
                                    self._logger.info(
                                        f"ADDING COMPONENT {component_instance.__class__} to {entity.type}")
                                    entity.add_component(component_instance)
                                else:
                                    self._logger.error(f"Class not found: {class_name}")

                    spawned_entities.append(entity)

            return spawned_entities

    # Desde config, Entities: Flora: 30
    # Bueno control de max, min etc. Primero pocos, 10 cada o asi, max, min 1.
    def __init__(self, env: simpyEnv, map: TileMap, flora_spawns: Spawns, fauna_spawns: Spawns):
        self._env: simpyEnv = env
        self._logger: Logger = logging.getLogger(Loggers.BIOME)
        self.spawn_system = WorldMapManager.SpawnSystem(env, flora_spawns, fauna_spawns)