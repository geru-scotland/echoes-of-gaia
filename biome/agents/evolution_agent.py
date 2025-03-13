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
from typing import List, Dict, Any

from biome.agents.base import Agent, TAction, TState
from biome.systems.evolution.genetics import GeneticAlgorithmModel
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.strings import Loggers
from shared.types import Observation
from utils.loggers import LoggerManager


class EvolutionAgentAI(Agent):
    def __init__(self, climate_data_manager: ClimateDataManager, entity_provider: EntityProvider):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self._genetic_model: GeneticAlgorithmModel = GeneticAlgorithmModel()

    def perceive(self) -> Observation:
        last_cycle_climate_data: List[Dict[str, Any]] = self._climate_data_manager.get_last_evolution_cycle_data()
        self._logger.error(f"PERCEIVING: {last_cycle_climate_data}")
        return {"data": last_cycle_climate_data}

    def decide(self, observation: TState) -> TAction:
        pass

    def act(self, action: TAction) -> None:
        pass