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
Módulo para analizar tendencias evolutivas en el bioma
"""
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np

from biome.entities.entity import Entity
from biome.systems.evolution.genetics import extract_genes_from_entity
from shared.enums.enums import EntityType


class EvolutionTrendAnalyzer:

    def __init__(self):
        self._flora_history = defaultdict(list)  # {species: [gen0_stats, gen1_stats, ...]}
        self._fauna_history = defaultdict(list)

        self._key_flora_traits = [
            "base_photosynthesis_efficiency",
            "cold_resistance",
            "heat_resistance",
            "optimal_temperature",
            "base_respiration_rate",
            "nutrient_absorption_rate"
        ]

        self._key_fauna_traits = [
            "cold_resistance",
            "heat_resistance",
            "optimal_temperature"
        ]

    def record_generation(self, entities: List[Entity], evolution_cycle: int) -> None:
        flora_stats = defaultdict(lambda: defaultdict(list))
        fauna_stats = defaultdict(lambda: defaultdict(list))

        for entity in entities:
            state_fields = entity.get_state_fields()
            entity_cycle = state_fields.get("general", {}).get("evolution_cycle", 0)

            if entity_cycle != evolution_cycle:
                continue

            try:
                genes = extract_genes_from_entity(entity)
                species = str(entity.get_species())

                if entity.get_type() == EntityType.FLORA:
                    for trait in self._key_flora_traits:
                        if hasattr(genes, trait):
                            flora_stats[species][trait].append(getattr(genes, trait))
                else:
                    for trait in self._key_fauna_traits:
                        if hasattr(genes, trait):
                            fauna_stats[species][trait].append(getattr(genes, trait))
            except Exception as e:
                print(f"Error al extraer genes: {e}")

        for species, traits in flora_stats.items():
            gen_avg = {trait: np.mean(values) for trait, values in traits.items() if values}
            if gen_avg:
                gen_avg["evolution_cycle"] = evolution_cycle
                self._flora_history[species].append(gen_avg)

        for species, traits in fauna_stats.items():
            gen_avg = {trait: np.mean(values) for trait, values in traits.items() if values}
            if gen_avg:
                gen_avg["evolution_cycle"] = evolution_cycle
                self._fauna_history[species].append(gen_avg)

    def analyze_trends(self) -> Dict[str, Any]:
        trends = {
            "flora": {},
            "fauna": {},
            "primary_adaptations": []
        }

        for species, history in self._flora_history.items():
            if len(history) < 2:  
                continue

            species_trends = {}

            for trait in self._key_flora_traits:
                values = [gen.get(trait, None) for gen in history]
                values = [v for v in values if v is not None]

                if len(values) < 2:
                    continue

                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]

                if trait == "optimal_temperature":
                    norm_factor = 80.0  # Rango de -30 a 50
                elif trait in ["cold_resistance", "heat_resistance"]:
                    norm_factor = 1.0
                else:
                    norm_factor = 1.0

                normalized_slope = slope / norm_factor
                species_trends[trait] = normalized_slope

            trends["flora"][species] = species_trends

        for species, history in self._fauna_history.items():
            if len(history) < 2:
                continue

            species_trends = {}

            for trait in self._key_fauna_traits:
                values = [gen.get(trait, None) for gen in history]
                values = [v for v in values if v is not None]

                if len(values) < 2:
                    continue

                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]

                if trait == "optimal_temperature":
                    norm_factor = 80.0
                elif trait in ["cold_resistance", "heat_resistance"]:
                    norm_factor = 1.0
                else:
                    norm_factor = 1.0

                normalized_slope = slope / norm_factor
                species_trends[trait] = normalized_slope

            trends["fauna"][species] = species_trends

        all_trends = []

        for entity_type in ["flora", "fauna"]:
            for species, traits in trends[entity_type].items():
                for trait, slope in traits.items():
                    all_trends.append((entity_type, species, trait, abs(slope), slope > 0))

        all_trends.sort(key=lambda x: x[3], reverse=True)

        strongest_trends = all_trends[:3]

        for entity_type, species, trait, magnitude, is_positive in strongest_trends:
            direction = "increasing" if is_positive else "decreasing"
            readable_trait = trait.replace("_", " ").title()
            trends["primary_adaptations"].append({
                "entity_type": entity_type,
                "species": species,
                "trait": readable_trait,
                "direction": direction,
                "magnitude": magnitude
            })

        return trends