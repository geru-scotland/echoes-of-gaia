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
from typing import Dict, Any

import simpy

from biome.biome import Biome
from biome.systems.evolution.registry import EvolutionAgentRegistry
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.managers.worldmap_manager import WorldMapManager
from shared.enums.enums import SimulationMode
from simulation.core.bootstrap.context.context_data import BiomeContextData


class BiomeAPI:
    def __init__(self, context: BiomeContextData, env: simpy.Environment, mode: SimulationMode,
                 options: Dict[str, Any] = None):
        self._biome = Biome(context, env, mode, options)

    def update(self, era: int, step: int):
        pass

    def get_biome(self) -> Biome:
        return self._biome

    def get_map_manager(self) -> WorldMapManager:
        return self._biome.get_worldmap_manager()

    def get_evolution_agent_registry(self) -> EvolutionAgentRegistry:
        return self._biome.get_evolution_agent_registry()

    def get_entity_provider(self) -> EntityProvider:
        return self._biome.get_entity_provider()
