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
from biome.systems.managers.entity_manager import EntityManager
from biome.systems.managers.worldmap_manager import WorldMapManager
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.state.handler import StateHandler
from shared.types import EntityList
from simulation.core.bootstrap.context.context_data import BiomeContextData
from simulation.core.systems.telemetry.datapoint import Datapoint


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
            self._entity_manager: EntityManager = EntityManager(self._map_manager.get_world_map())
            self._entity_collector: EntityDataCollector = EntityDataCollector(entity_manager=self._entity_manager)
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

    def collect_data(self, datapoint_id: int, timestamp: int) -> Datapoint:
        try:
            statistics: Dict[str, int|float] = self._entity_collector.collect_data()
            datapoint: Datapoint = Datapoint(
                measurement="biome_states_15",
                tags={"state_id": str(datapoint_id)},
                timestamp=timestamp,
                fields={**statistics}
            )

            self._logger.info("Creating datapoint...")
            self._logger.error(datapoint)
            self._logger.info("Collecting Biome data...")
            return datapoint
        except Exception as e:
            self._logger.exception(f"There was an error creating the Biome state datapoint: {e}")

    def compute_state(self):
        pass
