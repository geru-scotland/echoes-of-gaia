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
from typing import Protocol, Dict, Any

from shared.enums.strings import Loggers
from shared.types import Observation, SymbolicResult
from utils.loggers import LoggerManager


class SymbolicModuleInterface(Protocol):
    def infer(self, observation: Observation) -> SymbolicResult:
        ...

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        ...


class RuleBasedSymbolicModule:
    def __init__(self, rules_config: Dict[str, Any]):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.rules = rules_config
        self._logger.info(f"Symbolic module ready")

    def infer(self, symbolic_data: Observation) -> SymbolicResult:
        result = {}
        latest_data = self.data_service.get_latest_graph_data()

        if "species_data" in latest_data:
            for species, data in latest_data["species_data"].items():
                if data.get("population", 0) < 5:
                    result[f"{species}_status"] = "endangered"
                elif data.get("avg_stress", 0) > 75:
                    result[f"{species}_status"] = "stressed"

        return result

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        self.rules.update(new_rules)
