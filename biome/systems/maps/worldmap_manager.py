from shared.types import TileMap, Spawns


class WorldMapManager:

    class SpawnSystem:
        # TODO: spawn_flora, spawn_fauna
        # pasar info desde el config.
        def __int__(self):
            pass

    # Desde config, Entities: Flora: 30
    # Bueno control de max, min etc. Primero pocos, 10 cada o asi, max, min 1.
    def __init__(self, map: TileMap, flora_spawns: Spawns, fauna_spawns: Spawns):
        print(flora_spawns)
        print(fauna_spawns)