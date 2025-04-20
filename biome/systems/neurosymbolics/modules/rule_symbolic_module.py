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
from shared.types import Observation, SymbolicFeedback
from utils.loggers import LoggerManager
from shared.enums.enums import (
    SpeciesStatus, SpeciesAction, PredatorPreyBalance, EcosystemRisk,
    RecommendedAction, BiodiversityStatus, StabilityStatus,
    InterventionPriority, ClimateStatus, ClimateAction,
    MoistureStatus, MoistureAction
)


class SymbolicModuleInterface(Protocol):
    def infer(self, observation: Observation) -> SymbolicFeedback:
        ...

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        ...


class RuleBasedSymbolicModule:
    def __init__(self, rules_config: Dict[str, Any] = None):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.rules = rules_config or {}
        self._logger.info(f"Symbolic module ready")

    def infer(self, observation: Observation) -> SymbolicFeedback:
        result = {"context": {}}

        species_data = observation.get("species_data", {})
        biome_data = observation.get("biome_data", {})

        self._analyze_species_status(species_data, result)

        self._analyze_predator_prey_dynamics(biome_data, result)

        self._analyze_ecosystem_stability(biome_data, result)

        # self._analyze_climate_conditions(biome_data, result)

        return result

    def _analyze_species_status(self, species_data, result):
        for species, data in species_data.items():
            if data.get("population", 0) < 7:
                result[f"{species}_status"] = SpeciesStatus.ENDANGERED
                result[f"{species}_action"] = SpeciesAction.PROTECTION_NEEDED

            elif data.get("avg_stress", 0) > 75:
                result[f"{species}_status"] = SpeciesStatus.STRESSED
                result[f"{species}_action"] = SpeciesAction.STRESS_REDUCTION_NEEDED

            elif data.get("population", 0) > 40 and data.get("type") == "fauna":
                result[f"{species}_status"] = SpeciesStatus.OVERPOPULATED
                result[f"{species}_action"] = SpeciesAction.POPULATION_CONTROL_NEEDED

    def _analyze_predator_prey_dynamics(self, biome_data, result):
        predator_prey_ratio = biome_data.get("predator_prey_ratio", 0)

        result["predator_prey_balance"] = PredatorPreyBalance.PREDATOR_DOMINANT
        if predator_prey_ratio > 0.3:  # NOTA: TENIAS .05 al principio
            result["predator_prey_balance"] = PredatorPreyBalance.PREDATOR_DOMINANT
            result["ecosystem_risk"] = EcosystemRisk.PREY_EXTINCTION_RISK
            result["recommended_action"] = RecommendedAction.REDUCE_PREDATOR_PRESSURE
        elif predator_prey_ratio < 0.1 and biome_data.get("prey_population", 0) > 15:
            result["predator_prey_balance"] = PredatorPreyBalance.PREY_DOMINANT
            result["ecosystem_risk"] = EcosystemRisk.OVERPOPULATION_RISK
            result["recommended_action"] = RecommendedAction.INCREASE_PREDATOR_PRESSURE

    def _analyze_ecosystem_stability(self, neural_data, result):
        biodiversity = neural_data.get("biodiversity_index", 0)
        stability = neural_data.get("ecosystem_stability", 0)

        if biodiversity < 0.3:
            result["biodiversity_status"] = BiodiversityStatus.CRITICAL
            result["recommended_action"] = RecommendedAction.INTRODUCE_SPECIES_DIVERSITY

        if stability < 0.4:
            result["stability_status"] = StabilityStatus.UNSTABLE
            result["recommended_action"] = RecommendedAction.ECOSYSTEM_INTERVENTION_NEEDED
        elif stability > 0.8:
            result["stability_status"] = StabilityStatus.HIGHLY_STABLE
            result["intervention_priority"] = InterventionPriority.LOW

    def _analyze_climate_conditions(self, neural_data, result):
        temp = neural_data.get("temperature", 20)
        humidity = neural_data.get("humidity", 50)
        precipitation = neural_data.get("precipitation", 30)

        if temp < 0:
            result["climate_status"] = ClimateStatus.EXTREME_COLD
            result["climate_action"] = ClimateAction.INCREASE_TEMPERATURE
        elif temp > 35:
            result["climate_status"] = ClimateStatus.EXTREME_HEAT
            result["climate_action"] = ClimateAction.REDUCE_TEMPERATURE

        if humidity < 20:
            result["moisture_status"] = MoistureStatus.EXTREME_DRY
            result["moisture_action"] = MoistureAction.INCREASE_HUMIDITY
        elif humidity > 90 and precipitation > 70:
            result["moisture_status"] = MoistureStatus.EXTREME_WET
            result["moisture_action"] = MoistureAction.REDUCE_PRECIPITATION

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        self.rules.update(new_rules)
