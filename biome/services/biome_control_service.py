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
from typing import Optional, Any

from biome.agents.evolution_agent import EvolutionAgentAI
from biome.api.biome_api import BiomeAPI
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.evolution.registry import EvolutionAgentRegistry
from shared.enums.enums import FloraSpecies, FaunaSpecies
from shared.enums.events import BiomeEvent


class BiomeControlService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = BiomeControlService()
        return cls._instance

    def __init__(self):
        # OJO, API NO BIOMA En simulation cuando se cree api, inicializar. o BIOMECONTROL SELF en biome api init o asi
        self._api: BiomeAPI = None

    def initialize(self, api):
        self._api = api

    def spawn_entity(self, entity_class: Any, species_enum: FloraSpecies | FaunaSpecies, species_name: str, **kwargs):
        agent: EvolutionAgentAI = self._load_evolution_agent_registry(species_enum)
        current_generation: int = agent.get_current_generation() if agent else 0

        BiomeEventBus.trigger(
            BiomeEvent.CREATE_ENTITY,
            entity_class=entity_class,
            entity_species_enum=species_enum,
            species_name=str(species_name),
            evolution_cycle=current_generation
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
