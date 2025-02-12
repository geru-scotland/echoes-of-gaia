from shared.base import EnumBase


class Strings(EnumBase):
    BIOME_BUILDER = "biome_builder"
    SIMULATION_BUILDER = "simulation_builder"

    BIOME_CONTEXT = "biome_ctx"
    SIMULATION_CONTEXT = "simulation_ctx"

class Loggers(EnumBase):
    RENDER = "render"
    GAME = "game"
    SCENE = "scene"
    BIOME = "biome"
    SIMULATION = "simulation"