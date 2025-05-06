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
import sys
import traceback
from typing import Dict, Any, Type

import simpy

from biome.agents.base import Agent
from biome.agents.climate_agent import ClimateAgentAI
from biome.agents.equilibrium_agent import EquilibriumAgentAI
from biome.agents.evolution_agent import EvolutionAgentAI
from biome.agents.fauna_agent import FaunaAgentAI
from biome.environment import Environment
from biome.systems.climate.state import ClimateState
from biome.systems.climate.system import ClimateSystem
from biome.systems.components.registry import ComponentRegistry
from biome.systems.data.providers import BiomeDataProvider
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.evolution.registry import EvolutionAgentRegistry
from biome.systems.evolution.visualization.evo_crossover_tracker import GeneticCrossoverTracker
from biome.systems.evolution.visualization.evo_tracker import EvolutionTracker
from biome.systems.evolution.visualization.setup import setup_evolution_visualization_system
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.managers.worldmap_manager import WorldMapManager
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer, BiomeScoreResult
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from shared.enums.enums import Agents, AgentType, Season, WeatherEvent, FloraSpecies, BiomeType, FaunaSpecies, \
    EntityType, SimulationMode
from shared.enums.events import BiomeEvent
from shared.events.handler import EventHandler
from shared.timers import Timers
from shared.types import EntityList, Observation, EntityDefinitions
from simulation.core.bootstrap.context.context_data import BiomeContextData


class Biome(Environment, BiomeDataProvider, EventHandler):

    def __init__(self, context: BiomeContextData, env: simpy.Environment, mode: SimulationMode,
                 options: Dict[str, Any]):
        Environment.__init__(self, context, env)
        try:
            self._logger.info(self._context.config.get("type"))

            self._options: Dict[str, Any] = options
            cleanup_dead_entities: bool = self._options.get("remove_dead_entities")
            ComponentRegistry.initialize(env, cleanup_dead_entities)
            self._map_manager: WorldMapManager = WorldMapManager(self._env, tile_map=self._context.tile_map,
                                                                 flora_definitions=self._context.flora_definitions,
                                                                 fauna_definitions=self._context.fauna_definitions,
                                                                 remove_dead_entities=cleanup_dead_entities)

            self._entity_provider: EntityProvider = EntityProvider(self._map_manager.get_world_map())

            self._climate: ClimateSystem = ClimateSystem(self._context.biome_type, Season.SPRING)
            self._climate_data_manager = ClimateDataManager(self._env, self._climate)

            self._evolution_registry: EvolutionAgentRegistry = EvolutionAgentRegistry(self._climate_data_manager,
                                                                                      self._entity_provider)
            self._entity_collector: EntityDataCollector = EntityDataCollector(entity_provider=self._entity_provider,
                                                                              climate_manager=self._climate_data_manager,
                                                                              evolution_registry=self._evolution_registry)
            self._score_analyzer: BiomeScoreAnalyzer = BiomeScoreAnalyzer()

            self._climate.configure_record_callback(self._climate_data_manager.record_daily_data)

            self._agents: Dict[AgentType, Agent] = self._initialize_agents(mode)

            self._climate.set_entity_provider(self._entity_provider)
            self._env.process(self._run_climate_environmental_factors_update(Timers.Calendar.DAY))

            EventHandler.__init__(self)
            self._logger.info(f"Biome {self._context.biome_type} is ready!")
        except Exception as e:
            tb = traceback.format_exc()
            self._logger.exception(f"Error creating biome. Traceback: {tb}")
            self._logger.exception(f"There was an error creating the Biome: {e}")

    def _register_events(self):
        BiomeEventBus.register(BiomeEvent.CREATE_ENTITY, self._map_manager.add_entity)
        BiomeEventBus.register(BiomeEvent.REMOVE_ENTITY, self._map_manager.remove_entity)
        BiomeEventBus.register(BiomeEvent.ENTITY_DEATH, self._map_manager.handle_entity_death)
        BiomeEventBus.register(BiomeEvent.NEUROSYMBOLIC_SERVICE_READY, self._run_equilibrium_agent)

    def _initialize_agents(self, mode: SimulationMode) -> Dict[AgentType, Agent]:
        agents: Dict[AgentType, Agent] = {}

        climate_agent: ClimateAgentAI = ClimateAgentAI(self._climate, self._context.climate_model)
        agents.update({AgentType.CLIMATE_AGENT: climate_agent})
        self._env.process(self._run_agent(AgentType.CLIMATE_AGENT, Timers.Agents.Climate.CLIMATE_UPDATE))
        local_fov_config: Dict[str, Any] = self._context.config.get("map", {}).get("local_fov", {})

        if self._context.config.get("fauna_ia") and mode in (
                SimulationMode.NORMAL, SimulationMode.TRAINING_WITH_RL_MODEL, SimulationMode.UNTIL_EXTINCTION):
            fauna_agent: FaunaAgentAI = FaunaAgentAI(
                self._context.fauna_model,
                self._entity_provider,
                self._map_manager,
                self._context.biome_type,
                local_fov_config
            )
            agents.update({AgentType.FAUNA_AGENT: fauna_agent})
            self._env.process(self._run_agent(AgentType.FAUNA_AGENT, Timers.Agents.Fauna.FAUNA_UPDATE))

        evolution_tracker: EvolutionTracker = setup_evolution_visualization_system() if self._options.get(
            "evolution_tracking") else None
        crossover_tracker: GeneticCrossoverTracker = GeneticCrossoverTracker() if self._options.get(
            "crossover_tracking") else None

        self._initialize_evolution_agents(
            EntityType.FLORA,
            self._context.flora_definitions,
            FloraSpecies,
            evolution_tracker, crossover_tracker
        )

        self._initialize_evolution_agents(
            EntityType.FAUNA,
            self._context.fauna_definitions,
            FaunaSpecies,
            evolution_tracker, crossover_tracker
        )

        equilibrium_agent: EquilibriumAgentAI = EquilibriumAgentAI()
        agents.update({AgentType.EQUILIBRIUM_AGENT: equilibrium_agent})

        return agents

    def _initialize_evolution_agents(self, entity_type: EntityType, entity_definitions: EntityDefinitions,
                                     species_enum_class: Type[FloraSpecies | FaunaSpecies],
                                     evolution_tracker: EvolutionTracker,
                                     crossover_tracker: GeneticCrossoverTracker) -> None:
        type_name = "flora" if entity_type == EntityType.FLORA else "fauna"

        for entity_def in entity_definitions:
            try:
                species_name = entity_def.get("species", "").lower()
                species = species_enum_class(species_name)

                lifespan = entity_def.get("avg-lifespan", 5.0) * float(Timers.Calendar.YEAR)
                evolution_cycle_time = int(lifespan * 0.05)

                evolution_agent = EvolutionAgentAI(
                    self._climate_data_manager,
                    self._entity_provider,
                    entity_type,
                    species,
                    lifespan,
                    evolution_cycle_time,
                    self._evolution_registry,
                    smart_population=self._options.get("smart-population"),
                    smart_plot=self._options.get("smart_plot"),
                    evolution_tracker=evolution_tracker,
                    crossover_tracker=crossover_tracker,
                )

                self._evolution_registry.register_agent(species, evolution_agent)
                process = self._env.process(self._run_evolution_agent(species, evolution_cycle_time))
                self._evolution_registry.register_process(species, process)

                self._logger.info(
                    f"Created evolution agent for {type_name} species {species} with cycle: {evolution_cycle_time}")

            except ValueError as e:
                self._logger.warning(f"Invalid {type_name} species name. Error: {e}")

    def _run_evolution_agent(self, species: FloraSpecies | FaunaSpecies, delay: int):
        yield self._env.timeout(delay)

        while True:
            try:
                agent = self._evolution_registry.get_agent(species)
                # TODO: EVOLUTION CYCLE ha de ser incremental, hata un cap del 60% del lifetime
                # toma aquí el tiempo y vete incrementando progresivamente
                if agent:
                    self._logger.debug(f"Running evolution cycle for {species}, delay: {delay}")

                    observation = agent.perceive()
                    action = agent.decide(observation)
                    agent.act(action)
                    evolution_cycle_time: float = agent.get_evolution_cycle_time()

                    self._logger.debug(f"EVOLUTION AGENT. CURRENT EVOLUTION CYCLE TIME: {evolution_cycle_time}")
                    yield self._env.timeout(evolution_cycle_time)
                else:
                    self._logger.warning(f"Agent for species {species} not found!")

            except Exception as e:
                tb = traceback.format_exc()
                self._logger.exception(f"Error executing evolution for {species}: {e}. Traceback: {tb}")
                yield self._env.timeout(delay)

    def _run_agent(self, agent_type: AgentType, delay: int):
        yield self._env.timeout(delay)
        while True:
            try:
                # TODO, he qutiado tipado, hacer uno general para el agente
                agent: Agent = self._agents.get(agent_type, None)
                observation: Observation = agent.perceive()
                action = agent.decide(observation)
                agent.act(action)
                # TODO: delay basado en el weatherevent quizá?
                # echarle una pensada - si drought que persista por X días
                yield self._env.timeout(delay)
            except Exception as e:
                tb = traceback.format_exc()
                self._logger.exception(f"An exception ocurred running  agent: {e}. Traceback: {tb}")
                sys.exit(1)

    def _run_equilibrium_agent(self) -> None:
        equilibrium_agent: EquilibriumAgentAI = self._agents.get(AgentType.EQUILIBRIUM_AGENT, None)
        if equilibrium_agent:
            observation: Observation = equilibrium_agent.perceive()
            action = equilibrium_agent.decide(observation)
            equilibrium_agent.act(action)

    def _run_climate_environmental_factors_update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            self._climate.environmental_factors_update()
            yield self._env.timeout(delay)

    def get_entity_provider(self) -> EntityProvider:
        return self._entity_provider

    def get_world_map(self) -> WorldMap:
        return self._map_manager.get_world_map()

    def get_worldmap_manager(self) -> WorldMapManager:
        return self._map_manager

    def get_evolution_agent_registry(self) -> EvolutionAgentRegistry:
        return self._evolution_registry

    def get_entity_collector(self) -> EntityDataCollector:
        return self._entity_collector

    def get_climate_collector(self) -> ClimateDataCollector:
        return None

    def get_climate_system(self) -> ClimateState:
        return self._climate

    def get_biome_type(self) -> BiomeType:
        return self._context.biome_type

    def get_climate_data_manager(self) -> ClimateDataManager:
        return self._climate_data_manager

    def get_score_analyzer(self) -> BiomeScoreAnalyzer:
        return self._score_analyzer

    def compute_state(self):
        pass
