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
from logging import Logger
from typing import Any, Dict, Optional, List

from biome.agents.evolution_agent import EvolutionAgentAI
from shared.enums.enums import FloraSpecies
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class EvolutionAgentRegistry:
    def __init__(self, climate_data_manager, entity_provider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager = climate_data_manager
        self._entity_provider = entity_provider
        self._agents: Dict[FloraSpecies, EvolutionAgentAI] = {}
        self._evolution_processes: Dict[FloraSpecies, Any] = {}

    def register_agent(self, species: FloraSpecies, agent: EvolutionAgentAI) -> None:
        self._agents[species] = agent

    def get_agent(self, species: FloraSpecies) -> Optional[EvolutionAgentAI]:
        return self._agents.get(species)

    def register_process(self, species: FloraSpecies, process) -> None:
        self._evolution_processes[species] = process

    def get_all_species(self) -> List[FloraSpecies]:
        return list(self._agents.keys())