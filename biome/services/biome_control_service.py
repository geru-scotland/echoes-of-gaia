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
Singleton service for biome entity control and evolution management.

Handles entity spawning with genetic inheritance and lifecycle control;
manages evolution agent interactions and generation tracking.
Supports batch entity creation and evolution cycle adjustments - provides
centralized interface for programmatic biome manipulation operations.
"""

import random
import traceback
from typing import Optional, Any, List, Dict, Tuple

from biome.agents.evolution_agent import EvolutionAgentAI
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.evolution.registry import EvolutionAgentRegistry
from shared.enums.enums import FloraSpecies, FaunaSpecies, EntityType
from shared.enums.events import BiomeEvent
from shared.timers import Timers


class BiomeControlService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BiomeControlService()
        return cls._instance

    def __init__(self):
        self._api = None

    def initialize(self, api):
        self._api = api

    def spawn_entity(self, entity_class: Any, species_enum: FloraSpecies | FaunaSpecies,
                     species_name: str, clone_genes: bool = True, **kwargs):

        agent: EvolutionAgentAI = self._load_evolution_agent_registry(species_name)
        current_generation: int = agent.get_current_generation() if agent else 0

        custom_components = None
        entity_lifespan = agent.base_species_lifespan // Timers.Calendar.YEAR if agent else kwargs.get('lifespan', 10.0)

        if clone_genes and agent:
            components, lifespan = self._get_genes_from_existing_entity(
                agent,
                species_name,
                current_generation
            )

            if components is not None:
                custom_components = components

            if lifespan is not None:
                entity_lifespan = lifespan

        BiomeEventBus.trigger(
            BiomeEvent.CREATE_ENTITY,
            entity_class=entity_class,
            entity_species_enum=species_enum,
            species_name=str(species_name),
            lifespan=entity_lifespan,
            custom_components=custom_components,
            evolution_cycle=current_generation,
            control_service=True
        )

    def adjust_evolution_cycle(self, species: FloraSpecies | FaunaSpecies, adjustment_factor: float) -> bool:
        agent: EvolutionAgentAI = self._load_evolution_agent_registry(species)
        if agent:
            agent.adjust_evolution_cycle(adjustment_factor)
            return True

        return False

    def _load_evolution_agent_registry(self, species: FloraSpecies | FaunaSpecies) -> Optional[EvolutionAgentRegistry]:
        if not self._api:
            return False

        agent_registry: EvolutionAgentRegistry = self._api.get_evolution_agent_registry()
        agent = agent_registry.get_agent(species)
        return agent if agent else None

    def batch_spawn_entities(self, entity_class: Any, species_enum: FloraSpecies | FaunaSpecies,
                             species_name: str, count: int, clone_genes: bool = True, **kwargs):

        agent: EvolutionAgentAI = self._load_evolution_agent_registry(species_name)
        current_generation: int = agent.get_current_generation() if agent else 0

        custom_components = None
        entity_lifespan = agent.base_species_lifespan // Timers.Calendar.YEAR if agent else kwargs.get('lifespan', 10.0)

        if clone_genes and agent and count > 0:
            components, lifespan = self._get_genes_from_existing_entity(
                agent,
                species_name,
                current_generation
            )

            if components is not None:
                custom_components = components

            if lifespan is not None:
                entity_lifespan = lifespan

        for i in range(count):
            BiomeEventBus.trigger(
                BiomeEvent.CREATE_ENTITY,
                entity_class=entity_class,
                entity_species_enum=species_enum,
                species_name=str(species_name),
                lifespan=entity_lifespan,
                custom_components=custom_components,
                evolution_cycle=current_generation,
                control_service=True
            )

    def _get_genes_from_existing_entity(self, agent: EvolutionAgentAI,
                                        species_name: str,
                                        evolution_cycle: int) -> Tuple[Optional[List[Dict]], Optional[float]]:

        logger = agent._logger if hasattr(agent, '_logger') else None

        if logger:
            logger.debug(
                f"[GENE_CLONE] Looking for entity to clone genes - Species: {species_name}, Cycle: {evolution_cycle}")

        if not self._api:
            if logger:
                logger.warning("[GENE_CLONE] API not available to get entities")
            return None, None

        entity_provider = self._api.get_entity_provider()
        if not entity_provider:
            if logger:
                logger.warning("[GENE_CLONE] Entity provider not available")
            return None, None

        if agent.entity_type == EntityType.FLORA:
            entities = entity_provider.get_flora(only_alive=True)
            if logger:
                logger.debug(f"[GENE_CLONE] Found {len(entities)} living flora entities")
        else:
            entities = entity_provider.get_fauna(only_alive=True)
            if logger:
                logger.debug(f"[GENE_CLONE] Found {len(entities)} living fauna entities")

        matching_entities = [
            entity for entity in entities
            if str(entity.get_species()) == species_name and
               entity.get_state_fields().get("general", {}).get("evolution_cycle") == evolution_cycle
        ]

        if logger:
            logger.debug(
                f"[GENE_CLONE] Found {len(matching_entities)} entities of species {species_name} in cycle {evolution_cycle}")

        if not matching_entities:
            if logger:
                logger.warning(f"[GENE_CLONE] No entities found for species {species_name} in cycle {evolution_cycle}")
            return None, None

        selected_entity = random.choice(matching_entities)

        if logger:
            logger.debug(f"[GENE_CLONE] Selected entity ID: {selected_entity.get_id()} for gene cloning")

        entity_lifespan = selected_entity.lifespan

        if logger:
            logger.debug(f"[GENE_CLONE] Lifespan of source entity: {entity_lifespan}")

        try:
            from biome.systems.evolution.genetics import extract_genes_from_entity

            genes = extract_genes_from_entity(selected_entity)

            if logger:
                logger.debug(f"[GENE_CLONE] Genes successfully extracted for species {species_name}")

                log_msg = f"[GENE_CLONE] Gene values for entity {selected_entity.get_id()}:\n"

                # Hardcodeo esto por ahora, TODO: Quitar...
                common_attributes = [
                    "growth_modifier", "growth_efficiency", "max_size",
                    "max_vitality", "aging_rate", "health_modifier",
                    "cold_resistance", "heat_resistance", "optimal_temperature"
                ]

                for attr in common_attributes:
                    if hasattr(genes, attr):
                        log_msg += f"  - {attr}: {getattr(genes, attr)}\n"

                if agent.entity_type == EntityType.FLORA:
                    flora_attributes = [
                        "base_photosynthesis_efficiency", "base_respiration_rate",
                        "metabolic_activity", "nutrient_absorption_rate",
                        "mycorrhizal_rate", "base_nutritive_value", "base_toxicity"
                    ]

                    for attr in flora_attributes:
                        if hasattr(genes, attr):
                            log_msg += f"  - {attr}: {getattr(genes, attr)}\n"

                else:
                    fauna_attributes = [
                        "hunger_rate", "thirst_rate", "metabolism_efficiency",
                        "max_energy_reserves", "movement_energy_cost"
                    ]

                    for attr in fauna_attributes:
                        if hasattr(genes, attr):
                            log_msg += f"  - {attr}: {getattr(genes, attr)}\n"

                logger.debug(log_msg)

            components = genes.convert_genes_to_components()

            if logger:
                logger.debug(f"[GENE_CLONE] Number of generated components: {len(components)}")

                component_types = [comp.get("type", "Unknown") for comp in components]
                logger.debug(f"[GENE_CLONE] Component types: {component_types}")

            return components, entity_lifespan
        except Exception as e:
            if logger:
                logger.error(f"[GENE_CLONE] Error extracting genes from existing entity: {e}")
                logger.error(f"[GENE_CLONE] Traceback: {traceback.format_exc()}")
            return None, None
