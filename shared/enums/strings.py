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
from shared.enums.base import EnumBaseStr


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
    CLIMATE = "climate"
    RESEARCH = "research"
    BOOTSTRAP = "bootstrap"
    WORLDMAP = "world_map"
    TIME = "time"
    INFLUXDB = "influxdb"
    REINFORCEMENT = "reinforcement"
    CLIMATE_AGENT = "climate_agent"
    EVOLUTION_AGENT = "evolution_agent"
    FAUNA_AGENT = "fauna_agent"