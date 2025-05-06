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
import random

from shared.enums.enums import Interventions

from typing import Dict, Optional, Any

from biome.services.biome_control_service import BiomeControlService
from biome.systems.neurosymbolics.integrations.base_strategy import IntegrationStrategy
from shared.enums.enums import (
    SpeciesStatus, SpeciesAction, RecommendedAction,
    PredatorPreyBalance, EntityType, FloraSpecies, FaunaSpecies,
    StabilityStatusValue, EcosystemRiskValue, EvolutionFactor,
    PopulationThreshold, BiomassThreshold, PopulationRatio,
    PopulationTrend, SpawnCount, AmplificationFactor, StressThreshold
)
from shared.types import PredictionFeedback, SymbolicFeedback, IntegratedResult
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class NaiveWeightedIntegrationStrategy(IntegrationStrategy):
    def __init__(self):
        self._logger = LoggerManager.get_logger(Loggers.BIOME)
        self._control_service = BiomeControlService.get_instance()

        # TODO: Pasar a configs, lo pongo aqui para agilizar el desarrollo, pero es una cerdada.
        self._config = {
            "max_evolution_speedup": EvolutionFactor.ACCELERATE_EXTREME,
            "max_evolution_slowdown": EvolutionFactor.SLOW_EXTREME,
            "crisis_threshold": 0.3,
            "max_spawn_count": SpawnCount.MAXIMUM,
            "confidence_weights": {
                "neural": 0.3,
                "symbolic": 0.7
            },
            "biomass_threshold": BiomassThreshold.CRITICAL,
            "flora_spawn_threshold": PopulationThreshold.FLORA_SPAWN,
            "min_evolution_cycle_time": None,
            "max_evolution_cycle_time": None
        }

    def integrate(self,
                  neural_result: PredictionFeedback,
                  symbolic_result: SymbolicFeedback,
                  confidence_weights: Optional[Dict[str, float]] = None) -> IntegratedResult:
        if confidence_weights is None:
            confidence_weights = self._config["confidence_weights"]

        self._logger.info(f"Integrating neural and symbolic results with weights: {confidence_weights}")

        context: Dict[str, Any] = symbolic_result.get("context", {})

        integrated_result: IntegratedResult = {
            "predictions": neural_result,
            "recommendations": symbolic_result,
            "interventions": []
        }

        ecosystem_health = self._evaluate_ecosystem_health(neural_result, symbolic_result)

        self._analyze_biomass_state(neural_result, context, integrated_result)

        species_at_risk = self._identify_species_at_risk(symbolic_result)

        self._analyze_predictions_vs_current(neural_result, context, integrated_result)

        self._analyze_predator_prey_balance(neural_result, symbolic_result, context, integrated_result)

        for species, status in species_at_risk.items():
            intervention = self._generate_species_intervention(
                species,
                status,
                symbolic_result.get(f"{species}_action"),
                neural_result,
                ecosystem_health,
                context
            )

            if intervention:
                integrated_result["interventions"].append(intervention)

        if "graph_metrics" in symbolic_result:
            self._analyze_graph_insights(neural_result, symbolic_result,
                                         context, integrated_result)

        return integrated_result

    def _evaluate_ecosystem_health(self,
                                   neural_result: PredictionFeedback,
                                   symbolic_result: SymbolicFeedback) -> float:
        indicators = {}

        if 'ecosystem_stability' in neural_result:
            indicators['ecosystem_stability'] = neural_result['ecosystem_stability']

        if 'avg_stress' in neural_result:
            indicators['stress_health'] = 1.0 - (neural_result['avg_stress'] / 100.0)

        if 'stability_status' in symbolic_result:
            if isinstance(symbolic_result['stability_status'], str):
                status_value = StabilityStatusValue[symbolic_result['stability_status'].name]
            else:
                status_value = StabilityStatusValue[str(symbolic_result['stability_status'].name)]
            indicators['symbolic_stability'] = status_value

        if 'ecosystem_risk' in symbolic_result:
            if isinstance(symbolic_result['ecosystem_risk'], str):
                risk_value = EcosystemRiskValue[symbolic_result['ecosystem_risk'].name]
            else:
                risk_value = EcosystemRiskValue[str(symbolic_result['ecosystem_risk'].name)]
            indicators['symbolic_risk'] = risk_value

        graph_metrics = symbolic_result.get("graph_metrics", {})
        if "robustness" in graph_metrics:
            indicators['graph_robustness'] = graph_metrics["robustness"]

        if "connectivity" in graph_metrics:
            indicators['graph_connectivity'] = graph_metrics["connectivity"]

        if not indicators:
            return 0.5

        symbolic_indicators = {k: v for k, v in indicators.items() if k.startswith('symbolic_')}
        neural_indicators = {k: v for k, v in indicators.items() if not k.startswith('symbolic_')}

        weights = self._config["confidence_weights"]

        if symbolic_indicators and neural_indicators:
            symbolic_avg = sum(symbolic_indicators.values()) / len(symbolic_indicators)
            neural_avg = sum(neural_indicators.values()) / len(neural_indicators)
            return (symbolic_avg * weights["symbolic"] + neural_avg * weights["neural"]) / (
                    weights["symbolic"] + weights["neural"])
        elif symbolic_indicators:
            return sum(symbolic_indicators.values()) / len(symbolic_indicators)
        elif neural_indicators:
            return sum(neural_indicators.values()) / len(neural_indicators)
        else:
            return 0.5

    def _analyze_graph_insights(self,
                                neural_result: PredictionFeedback,
                                symbolic_result: SymbolicFeedback,
                                context: Dict[str, Any],
                                integrated_result: IntegratedResult) -> None:
        graph_metrics = symbolic_result.get("graph_metrics", {})
        if not graph_metrics:
            return

        keystone_species = symbolic_result.get("keystone_species", [])
        if keystone_species:
            available_species = context.get("available_species", {})

            for species in keystone_species:
                if species in available_species.get("flora", []) or species in available_species.get("fauna", []):
                    integrated_result["interventions"].append({
                        "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                        "species": species,
                        "factor": EvolutionFactor.ACCELERATE_SIGNIFICANT,
                        "reason": f"Protecting keystone species {species} based on network centrality"
                    })

        trophic_structure = symbolic_result.get("trophic_structure", {})
        if trophic_structure:
            levels_count = trophic_structure.get("levels_count", 0)

            if levels_count < 3 and "niche_diversity" in symbolic_result and symbolic_result[
                "niche_diversity"] == "low":
                predators = context.get("available_species", {}).get("predators", [])
                if predators:
                    integrated_result["interventions"].append({
                        "type": Interventions.SPAWN_ENTITIES,
                        "entity_type": EntityType.FAUNA,
                        "diet_type": "CARNIVORE",
                        "species": random.choice(predators),
                        "count": int(SpawnCount.MINIMAL),
                        "reason": "Diversifying trophic structure to increase ecosystem stability"
                    })

    def _identify_species_at_risk(self, symbolic_result: SymbolicFeedback) -> Dict[str, SpeciesStatus]:
        species_at_risk = {}

        for key, value in symbolic_result.items():
            if key.endswith("_status") and isinstance(value, SpeciesStatus):
                species_name = key.replace("_status", "")
                if value in [SpeciesStatus.ENDANGERED, SpeciesStatus.STRESSED, SpeciesStatus.OVERPOPULATED]:
                    species_at_risk[species_name] = value

        return species_at_risk

    def _analyze_biomass_state(self,
                               neural_result: PredictionFeedback,
                               context: Dict[str, Any],
                               integrated_result: IntegratedResult) -> None:
        current_state = context.get("current_state", {})
        available_species = context.get("available_species", {})

        flora = available_species.get("flora", [])
        herbivores = available_species.get("herbivores", [])
        current_biomass = current_state.get('biomass_density', 0.5)
        current_flora_count = current_state.get('flora_count', 20)
        herbivore_count = current_state.get('prey_population', 0)

        predicted_flora = neural_result.get('flora_count', current_flora_count)

        MIN_FLORA_PER_HERBIVORE = 5.0

        spawn_count = 0
        spawn_reason = ""

        # Check 1: Biomasa baja o el módulo neural me sugiere disminuir
        if (current_biomass < BiomassThreshold.LOW or
                (
                        predicted_flora < current_flora_count * 0.8 and current_flora_count < PopulationThreshold.FLORA_SPAWN)):
            flora_deficit = max(0, PopulationThreshold.FLORA_SPAWN - current_flora_count)
            spawn_count = flora_deficit
            spawn_reason = f"Low biomass density ({current_biomass:.2f}) or expected decline in flora"

        # Check 2: Proporción flora-herbívoro
        if herbivore_count > 0:
            current_ratio = current_flora_count / herbivore_count
            if current_ratio < MIN_FLORA_PER_HERBIVORE:
                flora_needed = int(herbivore_count * MIN_FLORA_PER_HERBIVORE - current_flora_count)

                if flora_needed > spawn_count:
                    spawn_count = flora_needed
                    spawn_reason = f"Flora-herbivore imbalance (current ratio: {current_ratio:.1f}, desired minimum: {MIN_FLORA_PER_HERBIVORE})"

        spawn_count = min(spawn_count, SpawnCount.MAXIMUM)

        if spawn_count > 0:
            if flora:
                integrated_result["interventions"].append({
                    "type": Interventions.SPAWN_ENTITIES,
                    "entity_type": EntityType.FLORA,
                    "species": random.choice(flora),
                    "count": int(spawn_count),
                    "reason": spawn_reason
                })

    def _analyze_predictions_vs_current(self,
                                        neural_result: PredictionFeedback,
                                        context: Dict[str, Any],
                                        integrated_result: IntegratedResult) -> None:

        current_state = context.get("current_state", {})
        available_species = context.get("available_species", {})

        current_prey = current_state.get('prey_population', 0)
        current_predator = current_state.get('predator_population', 0)
        current_flora = current_state.get('flora_count', 0)

        predicted_prey = round(neural_result.get('prey_population', current_prey))
        predicted_predator = round(neural_result.get('predator_population', current_predator))
        predicted_flora = round(neural_result.get('flora_count', current_flora))

        prey_trend = ((predicted_prey - current_prey) / max(1, current_prey)) if current_prey > 0 else 0
        predator_trend = (
                (predicted_predator - current_predator) / max(1, current_predator)) if current_predator > 0 else 0
        flora_trend = ((predicted_flora - current_flora) / max(1, current_flora)) if current_flora > 0 else 0

        self._logger.info(f"Prediction analysis - Current vs. Predicted:")
        self._logger.info(f"  Prey: {current_prey} → {predicted_prey} (trend: {prey_trend:.2%})")
        self._logger.info(f"  Predator: {current_predator} → {predicted_predator} (trend: {predator_trend:.2%})")
        self._logger.info(f"  Flora: {current_flora} → {predicted_flora} (trend: {flora_trend:.2%})")

        integrated_result["trends"] = {
            "prey": prey_trend,
            "predator": predator_trend,
            "flora": flora_trend
        }

    def _analyze_predator_prey_balance(self,
                                       neural_result: PredictionFeedback,
                                       symbolic_result: SymbolicFeedback,
                                       context: Dict[str, Any],
                                       integrated_result: IntegratedResult) -> None:

        current_state = context.get("current_state", {})
        available_species = context.get("available_species", {})

        current_prey = current_state.get('prey_population', 0)
        current_predator = current_state.get('predator_population', 0)
        current_flora = current_state.get('flora_count', 0)

        predicted_prey = round(neural_result.get('prey_population', current_prey))
        predicted_predator = round(neural_result.get('predator_population', current_predator))
        predicted_flora = round(neural_result.get('flora_count', current_flora))

        current_ratio = current_predator / max(1, current_prey) if current_prey > 0 else 0
        predicted_ratio = predicted_predator / max(1, predicted_prey) if predicted_prey > 0 else 0

        prey_trend = ((predicted_prey - current_prey) / max(1, current_prey)) if current_prey > 0 else 0
        predator_trend = (
                (predicted_predator - current_predator) / max(1, current_predator)) if current_predator > 0 else 0
        flora_trend = ((predicted_flora - current_flora) / max(1, current_flora)) if current_flora > 0 else 0

        self._logger.info(
            f"Population trends - Prey: {prey_trend:.2%}, Predator: {predator_trend:.2%}, Flora: {flora_trend:.2%}")

        amplification_factor = AmplificationFactor.NONE
        if abs(prey_trend) > PopulationTrend.MODERATE_INCREASE or abs(
                predator_trend) > PopulationTrend.MODERATE_INCREASE:
            amplification_factor = AmplificationFactor.SIGNIFICANT
            self._logger.info(f"Neural predictions indicate significant population changes. Amplifying interventions.")

        predator_prey_balance = symbolic_result.get('predator_prey_balance')

        # ====================== CASOS SIMBÓLICOS ======================
        if predator_prey_balance == PredatorPreyBalance.PREDATOR_DOMINANT:
            # Situación: demasiados depredadores en relación a las presas

            # 1. Acelero evolución de presas para que se adapten a la presión de predators
            preys = available_species.get("preys", [])
            if preys:
                integrated_result["interventions"].append({
                    "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                    "target": "prey",
                    "species": random.choice(preys),
                    "factor": max(EvolutionFactor.ACCELERATE_EXTREME,
                                  EvolutionFactor.ACCELERATE_MODERATE / amplification_factor),
                    "reason": "Accelerating prey evolution to adapt to predator pressure"
                })

            # 2. Ralentizo evolución de depredadores
            predators = available_species.get("predators")
            if predators:
                integrated_result["interventions"].append({
                    "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                    "target": "predator",
                    "species": random.choice(predators),
                    "factor": min(EvolutionFactor.SLOW_SIGNIFICANT,
                                  EvolutionFactor.SLOW_MODERATE * amplification_factor),
                    "reason": "Slowing predator evolution to reduce pressure on prey"
                })

            # 3. Si las presas en niveles críticos, spawneo nuevas (por ahora, hago clones de max gen)
            if current_prey <= PopulationThreshold.PREY_CRITICAL:
                herbivores = available_species.get("herbivores", [])
                if herbivores:
                    base_count = min(int(SpawnCount.MEDIUM), int(SpawnCount.MAXIMUM))
                    spawn_count = self._calculate_spawn_count(base_count, "HERBIVORE", current_state)
                    integrated_result["interventions"].append({
                        "type": Interventions.SPAWN_ENTITIES,
                        "entity_type": EntityType.FAUNA,
                        "diet_type": "HERBIVORE",
                        "species": random.choice(herbivores),
                        "count": spawn_count,
                        "reason": "Introducing prey to prevent extinction due to predator dominance"
                    })

        elif predator_prey_balance == PredatorPreyBalance.PREY_DOMINANT:

            # Hago parecido al anterior
            predators = available_species.get("predators")
            if predators:
                integrated_result["interventions"].append({
                    "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                    "target": "predator",
                    "species": random.choice(predators),
                    "factor": max(EvolutionFactor.ACCELERATE_EXTREME,
                                  EvolutionFactor.ACCELERATE_SIGNIFICANT / amplification_factor),
                    "reason": "Accelerating predator evolution to control prey population"
                })

            if current_predator < PopulationThreshold.PREDATOR_CRITICAL or (
                    current_prey > PopulationThreshold.PREY_ABUNDANT and current_predator < SpawnCount.LOW):
                base_spawn_count = int(SpawnCount.LOW)
                if predator_trend < 0:
                    base_spawn_count = min(int(SpawnCount.MEDIUM), int(SpawnCount.MAXIMUM))

                carnivores = available_species.get("carnivores", [])
                if carnivores:
                    spawn_count = self._calculate_spawn_count(base_spawn_count, "CARNIVORE", current_state)
                    integrated_result["interventions"].append({
                        "type": Interventions.SPAWN_ENTITIES,
                        "entity_type": EntityType.FAUNA,
                        "diet_type": "CARNIVORE",
                        "species": random.choice(carnivores),
                        "count": spawn_count,
                        "reason": "Introducing predators to control excessive prey population"
                    })

        # ====================== CASOS HÍBRIDOS ======================

        # CASO 1: Si ratio crítico y predicciones negativas para presas
        if current_ratio > PopulationRatio.PREDATOR_DOMINANT and prey_trend < PopulationTrend.MODERATE_DECLINE:
            herbivores = available_species.get("herbivores", [])
            if herbivores:
                integrated_result["interventions"].append({
                    "type": Interventions.SPAWN_ENTITIES,
                    "entity_type": EntityType.FAUNA,
                    "diet_type": "HERBIVORE",
                    "species": random.choice(herbivores),
                    "count": int(SpawnCount.MEDIUM),
                    "reason": "Hybrid decision: High predator/prey ratio and predicted prey decline"
                })

        # CASO 2: Ratio muy bajo y predicciones positivas para presas
        if current_ratio < PopulationRatio.PREY_DOMINANT and prey_trend > PopulationTrend.SIGNIFICANT_INCREASE and current_prey > PopulationThreshold.PREY_ABUNDANT:
            carnivores = available_species.get("carnivores", [])
            if carnivores:
                base_count = int(SpawnCount.MINIMAL)
                spawn_count = self._calculate_spawn_count(base_count, "CARNIVORE", current_state)

                integrated_result["interventions"].append({
                    "type": Interventions.SPAWN_ENTITIES,
                    "entity_type": EntityType.FAUNA,
                    "diet_type": "CARNIVORE",
                    "species": spawn_count,
                    "count": int(SpawnCount.MINIMAL),
                    "reason": "Hybrid decision: Low predator/prey ratio and predicted prey increase"
                })

        # CASO 3: Cambio drástico predicho en el ratio
        if current_ratio > 0 and predicted_ratio > 0 and abs(predicted_ratio - current_ratio) / current_ratio > 0.25:
            if predicted_ratio > current_ratio:
                preys = available_species.get("preys")
                if preys:
                    integrated_result["interventions"].append({
                        "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                        "target": "prey",
                        "species": random.choice(preys),
                        "factor": EvolutionFactor.ACCELERATE_MODERATE,
                        "reason": "Hybrid decision: Neural network predicts significant increase in predator/prey ratio"
                    })
            else:
                predators = available_species.get("predators")
                if predators:
                    integrated_result["interventions"].append({
                        "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                        "target": "predator",
                        "species": random.choice(predators),
                        "factor": EvolutionFactor.ACCELERATE_MODERATE,
                        "reason": "Hybrid decision: Neural network predicts significant decrease in predator/prey ratio"
                    })

        # CASO 4: Población crítica de flora con predicción negativa
        if current_flora < PopulationThreshold.FLORA_CRITICAL and flora_trend < PopulationTrend.MODERATE_DECLINE:
            flora = available_species.get("flora", [])
            if flora:
                integrated_result["interventions"].append({
                    "type": Interventions.SPAWN_ENTITIES,
                    "entity_type": EntityType.FLORA,
                    "species": random.choice(flora),
                    "count": int(SpawnCount.MEDIUM),
                    "reason": "Hybrid decision: Low flora count and predicted flora decline"
                })

        # CASO 5: Si se predice aumento en estrés
        if 'avg_stress' in neural_result and neural_result['avg_stress'] > current_state.get('avg_stress',
                                                                                             0) + StressThreshold.SIGNIFICANT_INCREASE:
            species_stress = symbolic_result.get('species_stress', {})
            if species_stress:
                most_stressed_species = max(species_stress.items(), key=lambda x: x[1])[0]
                integrated_result["interventions"].append({
                    "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                    "species": most_stressed_species,
                    "factor": EvolutionFactor.ACCELERATE_SIGNIFICANT,
                    "reason": f"Hybrid decision: Neural network predicts significant stress increase for {most_stressed_species}"
                })

    def _generate_species_intervention(self,
                                       species_name: str,
                                       status: SpeciesStatus,
                                       action: SpeciesAction,
                                       neural_result: PredictionFeedback,
                                       ecosystem_health: float,
                                       context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current_state = context.get("current_state", {})
        if status == SpeciesStatus.ENDANGERED:
            is_flora = any(s == species_name for s in FloraSpecies)
            entity_type = EntityType.FLORA if is_flora else EntityType.FAUNA

            diet_type = None
            if not is_flora:
                diet_type = "HERBIVORE"
                if any(s == species_name for s in context.get("available_species", {}).get("carnivores", [])):
                    diet_type = "CARNIVORE"
                elif any(s == species_name for s in context.get("available_species", {}).get("omnivores", [])):
                    diet_type = "OMNIVORE"

            base_spawn_count = random.randint(int(SpawnCount.LOW), int(SpawnCount.MAXIMUM))

            if not is_flora:
                spawn_count = self._calculate_spawn_count(base_spawn_count, diet_type, current_state)
            else:
                spawn_count = base_spawn_count

            return {
                "type": Interventions.SPAWN_ENTITIES,
                "entity_type": entity_type,
                "species": species_name,
                "count": spawn_count,
                "reason": f"Species {species_name} is endangered"
            }

        elif status == SpeciesStatus.STRESSED:
            adaptation_factor = max(
                EvolutionFactor.ACCELERATE_EXTREME,
                EvolutionFactor.ACCELERATE_SIGNIFICANT + ((1 - ecosystem_health) * 0.3)
            )

            return {
                "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                "species": species_name,
                "factor": adaptation_factor,
                "reason": f"Accelerating evolution to adapt to stress"
            }

        elif status == SpeciesStatus.OVERPOPULATED:
            return {
                "type": Interventions.ADJUST_EVOLUTION_CYCLE,
                "species": species_name,
                "factor": EvolutionFactor.SLOW_EXTREME,
                "reason": f"Slowing evolution to reduce overpopulation"
            }

        return None

    def _calculate_spawn_count(self,
                               base_count: int,
                               diet_type: str,
                               current_state: Dict[str, Any]) -> int:
        diet_factors = {
            "CARNIVORE": 0.3,
            "OMNIVORE": 0.5,
            "HERBIVORE": 1.0,
            None: 0.7
        }

        factor = diet_factors.get(diet_type, diet_factors[None])

        current_prey = current_state.get('prey_population', 0)
        current_predator = current_state.get('predator_population', 0)

        if diet_type in ["CARNIVORE", "OMNIVORE"] and current_prey > 0:
            predator_prey_ratio = current_predator / current_prey

            if predator_prey_ratio > 0.5:
                factor *= 0.5
            elif predator_prey_ratio < 0.1:
                factor *= 1.5

        adjusted_count = max(1, int(base_count * factor))

        return adjusted_count
