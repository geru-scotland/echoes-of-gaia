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

from biome.agents.base import Agent
from biome.agents.climate_agent import ClimateAgentAI
from biome.environment import Environment
from biome.systems.climate.state import ClimateState
from biome.systems.climate.system import ClimateSystem
from biome.systems.data.providers import BiomeDataProvider
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.managers.worldmap_manager import WorldMapManager
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer, BiomeScoreResult
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.state.handler import StateHandler
from shared.enums.enums import Agents, AgentType, Season, WeatherEvent
from shared.enums.events import BiomeEvent
from shared.events.handler import EventHandler
from shared.timers import Timers
from shared.types import EntityList, Observation
from simulation.core.bootstrap.context.context_data import BiomeContextData
from simulation.core.systems.telemetry.datapoint import Datapoint


class Biome(Environment, StateHandler, BiomeDataProvider, EventHandler):

    def __init__(self, context: BiomeContextData, env: simpy.Environment):
        super().__init__(context, env)
        try:
            self._logger.info(self._context.config.get("type"))
            self._map_manager: WorldMapManager = WorldMapManager(self._env, tile_map=self._context.tile_map,
                                                                 flora_definitions=self._context.flora_definitions,
                                                                 fauna_definitions=self._context.fauna_definitions)
            self._entity_provider: EntityProvider = EntityProvider(self._map_manager.get_world_map())
            self._entity_collector: EntityDataCollector = EntityDataCollector(entity_provider=self._entity_provider)
            self._score_analyzer: BiomeScoreAnalyzer = BiomeScoreAnalyzer()
            self._agents: Dict[AgentType, Agent] = self._initialize_agents()

            self._logger.info("Biome is ready!")
        except Exception as e:
            self._logger.exception(f"There was an error creating the Biome: {e}")

    def _initialize_agents(self) -> Dict[AgentType, Agent]:
        agents: Dict[AgentType, Agent] = {}

        climate: ClimateSystem = ClimateSystem(self._context.biome_type, Season.SPRING)
        climate_agent: ClimateAgentAI = ClimateAgentAI(climate, self._context.climate_model)

        agents.update({AgentType.CLIMATE_AGENT: climate_agent})
        self._env.process(self._run_climate_agent(Timers.Agents.CLIMATE_UPDATE))
        return agents

    def _run_climate_agent(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            try:
                agent = self._agents.get(AgentType.CLIMATE_AGENT)
                observation: Observation = agent.perceive()
                action: WeatherEvent = agent.decide(observation)
                agent.act(action)
                # TODO: delay basado en el weatherevent quizá?
                # echarle una pensada - si drought que persista por X días
                yield self._env.timeout(delay)
            except Exception as e:
                self._logger.exception(f"An exception ocurred running climate agent: {e}")

    def _register_events(self):
        BiomeEventBus.register(BiomeEvent.CREATE_ENTITY, self._map_manager.add_entity)
        BiomeEventBus.register(BiomeEvent.REMOVE_ENTITY, self._map_manager.remove_entity)

    def update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            self._logger.info(f"BIOMA UPDATE!... t={self._env.now}")
            yield self._env.timeout(25)

    def resolve_pending_components(self):
        self._logger.info("Resolving pending components...")

    def get_entity_provider(self) -> EntityProvider:
        return self._entity_provider

    def get_world_map(self) -> WorldMap:
        return self._map_manager.get_world_map()

    def get_entity_collector(self) -> EntityDataCollector:
        return self._entity_collector

    def get_score_analyzer(self) -> BiomeScoreAnalyzer:
        return self._score_analyzer

    def compute_state(self):
        pass
