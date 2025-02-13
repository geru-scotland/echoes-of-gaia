from shared.base import EnumBaseStr


class Strings(EnumBaseStr):
    BIOME_BUILDER = "biome_builder"
    SIMULATION_BUILDER = "simulation_builder"

    BIOME_CONTEXT = "biome_ctx"
    SIMULATION_CONTEXT = "simulation_ctx"

class Loggers(EnumBaseStr):
    RENDER = "render"
    GAME = "game"
    SCENE = "scene"
    BIOME = "biome"
    SIMULATION = "simulation"