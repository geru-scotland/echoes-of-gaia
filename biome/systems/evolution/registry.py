""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""

"""
Evolution agent registry for coordinating species evolution processes.

Maintains references to evolution agents organized by species type;
manages process registration and inter-agent communication.
Tracks evolution trends and provides generational analysis capabilities.
"""

from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

from biome.agents.evolution_agent import EvolutionAgentAI
from biome.systems.evolution.trend_analyzer import EvolutionTrendAnalyzer
from shared.enums.enums import EntityType, FaunaSpecies, FloraSpecies
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class EvolutionAgentRegistry:
    def __init__(self, climate_data_manager, entity_provider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager = climate_data_manager
        self._entity_provider = entity_provider

        self._flora_agents: Dict[FloraSpecies, EvolutionAgentAI] = {}
        self._fauna_agents: Dict[FaunaSpecies, EvolutionAgentAI] = {}

        self._flora_processes: Dict[FloraSpecies, Any] = {}
        self._fauna_processes: Dict[FaunaSpecies, Any] = {}
        self._trend_analyzer = EvolutionTrendAnalyzer()

    def register_agent(self, species: FloraSpecies | FaunaSpecies, agent: EvolutionAgentAI) -> None:
        if agent.entity_type == EntityType.FLORA:
            self._flora_agents[species] = agent
        else:
            self._fauna_agents[species] = agent

    def get_agent(self, species: FloraSpecies | FaunaSpecies) -> Optional[EvolutionAgentAI]:
        if species in self._flora_agents:
            return self._flora_agents[species]
        elif species in self._fauna_agents:
            return self._fauna_agents[species]
        return None

    def register_process(self, species: FloraSpecies | FaunaSpecies, process) -> None:
        if species in self._flora_agents:
            self._flora_processes[species] = process
        elif species in self._fauna_agents:
            self._fauna_processes[species] = process

    def get_all_species(self) -> Tuple[List[FloraSpecies], List[FaunaSpecies]]:
        return list(self._flora_agents.keys()), list(self._fauna_agents.keys())

    def record_generation(self, evolution_cycle: int) -> None:
        all_flora = self._entity_provider.get_flora(only_alive=True)
        all_fauna = self._entity_provider.get_fauna(only_alive=True)
        all_entities = all_flora + all_fauna

        self._trend_analyzer.record_generation(all_entities, evolution_cycle)

    def analyze_evolution_trends(self) -> Dict[str, Any]:
        return self._trend_analyzer.analyze_trends()