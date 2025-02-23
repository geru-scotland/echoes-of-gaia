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

from biome.components.biome.climate import Climate
from biome.environment import Environment
from biome.systems.maps.manager import WorldMapManager
from biome.systems.state.handler import StateHandler
from shared.types import BiomeStateData
from simulation.core.bootstrap.context.context_data import BiomeContextData


class Biome(Environment, StateHandler):
    def __init__(self, context: BiomeContextData, env: simpy.Environment):
        super().__init__(context, env)
        try:
            # Del contexto, habrá que pasar datos de clima de los config
            self._env.process(self.update(25))
            self.add_component(Climate(self._env))
            self._logger.info(self._context.config.get("type"))
            self._map_manager: WorldMapManager = WorldMapManager(self._env, tile_map=self._context.tile_map,
                                                                 flora_definitions=self._context.flora_definitions,
                                                                 fauna_definitions=self._context.fauna_definitions)
            self._logger.info("Biome is ready!")
        except Exception as e:
            self._logger.exception(f"There was an error creating the Biome: {e}")

    def update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            self._logger.info(f"BIOMA UPDATE!... t={self._env.now}")
            yield self._env.timeout(25)

    def resolve_pending_components(self):
        self._logger.info("Resolving pending components...")

    def collect_data(self) -> BiomeStateData:
        data: BiomeStateData = {}
        self._logger.info("Creating datapoint...")
        self._logger.info("Collecting Biome data...")
        return data

    def compute_state(self):
        pass
