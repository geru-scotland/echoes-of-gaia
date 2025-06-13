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
Neurosymbolic integration balancer for biome intervention decisions.

Combines neural and symbolic module outputs with confidence weighting;
processes observations and applies evolution-spawning interventions.
Handles species availability analysis and intervention execution via
biome control service - supports adaptive biome management strategies.
"""

from logging import Logger
from typing import Any, Dict, List, Optional, Type

from biome.services.biome_control_service import BiomeControlService
from biome.systems.neurosymbolics.data_service import NeurosymbolicDataService
from biome.systems.neurosymbolics.integrations.base_strategy import IntegrationStrategy
from biome.systems.neurosymbolics.modules.neural_module import NeuralModuleInterface
from biome.systems.neurosymbolics.modules.rule_symbolic_module import (
    SymbolicModuleInterface,
)
from shared.enums.enums import (
    DietType,
    EntityType,
    FaunaSpecies,
    FloraSpecies,
    Interventions,
)
from shared.enums.strings import Loggers
from shared.types import (
    IntegratedResult,
    Observation,
    PredictionFeedback,
    SymbolicFeedback,
)
from utils.loggers import LoggerManager


class NeuroSymbolicBalancer:
    def __init__(self,
                 neural_module: Type[NeuralModuleInterface],
                 symbolic_module: Type[SymbolicModuleInterface],
                 integration_strategy: Type[IntegrationStrategy]):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initialising NeurySymbolic balancer...")
        self.neural_module: NeuralModuleInterface = neural_module()
        self.symbolic_module: SymbolicModuleInterface = symbolic_module()
        self.integration_strategy: IntegrationStrategy = integration_strategy()
        self.confidence_weights: Dict[str, float] = {"neural": 0.6, "symbolic": 0.4}

    def get_observation(self, data_service: NeurosymbolicDataService) -> Observation:
        neural_sequence = data_service.get_neural_sequence()
        graph_data = data_service.get_graph_data()

        return {
            'neural_data': neural_sequence,
            'graph_data': graph_data
        }

    def process(self, observation: Observation) -> Optional[IntegratedResult]:
        neural_data = observation.get("neural_data", {})

        graph_data = observation.get("graph_data", {})

        current_state = {}
        if "biome_data" in graph_data:
            current_state = graph_data["biome_data"]
            current_state["flora_count"] = neural_data[-1, 2]

        neural_feedback: PredictionFeedback = self.neural_module.predict(neural_data)

        symbolic_feedback: SymbolicFeedback = self.symbolic_module.infer(graph_data)
        symbolic_feedback["context"]["current_state"] = current_state
        symbolic_feedback["context"]["available_species"] = self._get_available_species(graph_data["species_data"])

        self._logger.debug(f"[NeuroSymbolic Balancer] Neural feedback: {neural_feedback}")
        self._logger.debug(f"[NeuroSymbolic Balancer] Symbolic feedback: {symbolic_feedback}")

        integrated_result = self.integration_strategy.integrate(
            neural_feedback,
            symbolic_feedback,
            self.confidence_weights
        )

        return integrated_result

    def apply_interventions(self, integrated_result: IntegratedResult) -> None:
        if not integrated_result or "interventions" not in integrated_result:
            self._logger.warning("No interventions to apply")
            return

        interventions = integrated_result.get("interventions", [])
        self._logger.debug(f"Applying {len(interventions)} interventions to the biome")

        for intervention in interventions:
            self._apply_intervention(intervention)

    def _get_available_species(self, species_data: Dict[str, Any]) -> Dict[str, Any]:
        available_species: Dict[str, List[FloraSpecies | FaunaSpecies]] = {
            "flora": [key for key, value in species_data.items()
                      if value.get("type", None) == EntityType.FLORA],
            "fauna": [key for key, value in species_data.items()
                      if value.get("type", None) == EntityType.FAUNA],
            "predators": [key for key, value in species_data.items()
                          if value.get("type", None) == EntityType.FAUNA and
                          value.get("diet", None) == DietType.CARNIVORE.value],
            "preys": [key for key, value in species_data.items()
                      if value.get("type", None) == EntityType.FAUNA and
                      value.get("diet", None) == DietType.HERBIVORE.value]
        }

        return available_species

    def _apply_intervention(self, intervention: Dict[str, Any]) -> None:
        intervention_type = intervention.get("type")
        reason = intervention.get("reason", "No reason provided")

        self._logger.debug(f"Applying intervention: {intervention_type} - Reason: {reason}")

        try:
            if intervention_type == Interventions.ADJUST_EVOLUTION_CYCLE:
                species: FloraSpecies | FaunaSpecies = intervention.get("species", None)
                if not species:
                    self._logger.warning(f"Missing species/target in evolution adjustment: {intervention}")
                    return

                factor = intervention.get("factor", 1.0)
                self._logger.debug(f"Adjusting evolution cycle for {species} with factor {factor}")
                BiomeControlService.get_instance().adjust_evolution_cycle(species, factor)

            elif intervention_type == Interventions.SPAWN_ENTITIES:
                self._handle_spawn_intervention(intervention)

        except Exception as e:
            self._logger.error(f"Error applying intervention: {e}")

    def _handle_spawn_intervention(self, intervention: Dict[str, Any]) -> None:

        entity_type = intervention.get("entity_type")
        species_name = intervention.get("species")
        count = intervention.get("count", 1)
        diet_type = intervention.get("diet_type")
        copy_genes = intervention.get("copy_genes", False)

        entity_class = None
        species_enum = None

        if entity_type == EntityType.FLORA:
            from biome.entities.flora import Flora
            entity_class = Flora

            if species_name:
                try:
                    species_enum = FloraSpecies
                except ValueError:
                    self._logger.warning(f"Unknown flora species: {species_name}")
                    return

        elif entity_type == EntityType.FAUNA:
            from biome.entities.fauna import Fauna
            entity_class = Fauna

            if species_name:
                try:
                    species_enum = FaunaSpecies
                except ValueError:
                    self._logger.warning(f"Unknown fauna species: {species_name}")
                    return

        if entity_class and species_enum:
            self._logger.debug(f"Spawning {count} {species_name} entities" +
                               (f" with gene cloning" if copy_genes else ""))

            kwargs = {}
            if entity_type == EntityType.FAUNA and diet_type:
                kwargs["diet_type"] = diet_type

            BiomeControlService.get_instance().batch_spawn_entities(
                entity_class=entity_class,
                species_enum=species_enum,
                species_name=str(species_name),
                count=count,
                clone_genes=copy_genes,
                **kwargs
            )
