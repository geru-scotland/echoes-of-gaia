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
import itertools
from logging import Logger
from typing import List, Dict, Any

from pandas import DataFrame

from biome.agents.base import Agent, TAction, TState
from biome.systems.evolution.genetics import GeneticAlgorithmModel
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.strings import Loggers
from shared.types import Observation, EntityList
from utils.loggers import LoggerManager


class EvolutionAgentAI(Agent):
    def __init__(self, climate_data_manager: ClimateDataManager, entity_provider: EntityProvider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self.entity_provider: EntityProvider = entity_provider
        self._genetic_model: GeneticAlgorithmModel = GeneticAlgorithmModel()
        self._evolution_cycle: itertools.count[int] = itertools.count(0)
        self._current_evolution_cycle: int = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)


    def perceive(self) -> Observation:
        return {
            "climate_data": self._climate_data_manager.get_data(self._current_evolution_cycle)[-1:],
            "flora": self.entity_provider.get_flora()
        }

    def decide(self, observation: Observation) -> TAction:
        pass

    def act(self, action: TAction) -> None:
        self._current_evolution_cycle = next(self._evolution_cycle)
        self._climate_data_manager.set_evolution_cycle(self._current_evolution_cycle)
