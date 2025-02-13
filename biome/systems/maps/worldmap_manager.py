from simpy import Environment as simpyEnv

from biome.components.entities.growth import GrowthComponent
from biome.entities.flora import Flora
from shared.stores.biome_store import BiomeStore
from shared.types import TileMap, Spawns


class WorldMapManager:

    class SpawnSystem:
        # TODO: spawn_flora, spawn_fauna
        # pasar info desde el config.
        def __init__(self):
            pass

    # Desde config, Entities: Flora: 30
    # Bueno control de max, min etc. Primero pocos, 10 cada o asi, max, min 1.
    def __init__(self, env: simpyEnv, map: TileMap, flora_spawns: Spawns, fauna_spawns: Spawns):
        self._env: simpyEnv = env
        spawn = flora_spawns[0]
        print(spawn)
        data = BiomeStore.flora.get(spawn.get("type"))
        flora = Flora(env)
        flora.add_component(GrowthComponent(env, data.get("growth_rate")))