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
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Any, TypedDict, Union
from dataclasses import dataclass, field

from shared.enums.enums import EvolutionSummary, TraitRecord


@dataclass
class EvolutionTracker:
    species_data: Dict[str, Dict[str, List[TraitRecord]]] = field(default_factory=dict)

    fitness_data: Dict[str, Dict[int, float]] = field(default_factory=dict)

    population_data: Dict[str, Dict[int, int]] = field(default_factory=dict)

    tracked_traits: Set[str] = field(default_factory=set)

    current_generation: Dict[str, int] = field(default_factory=dict)

    def register_trait_values(self,
                              species: str,
                              generation: int,
                              traits_values: Dict[str, float]) -> None:

        if species not in self.species_data:
            self.species_data[species] = {}
            self.current_generation[species] = generation

        self.current_generation[species] = max(self.current_generation.get(species, 0), generation)

        for trait, value in traits_values.items():
            if trait not in self.species_data[species]:
                self.species_data[species][trait] = []

            self.species_data[species][trait].append(TraitRecord(generation, value))
            self.tracked_traits.add(trait)

    def register_population(self,
                            species: str,
                            generation: int,
                            count: int) -> None:

        if species not in self.population_data:
            self.population_data[species] = {}

        self.population_data[species][generation] = count

    def register_fitness(self,
                         species: str,
                         generation: int,
                         fitness: float) -> None:

        if species not in self.fitness_data:
            self.fitness_data[species] = {}

        self.fitness_data[species][generation] = fitness

    def get_trait_trend(self, species: str, trait: str) -> Optional[float]:

        if species not in self.species_data or trait not in self.species_data[species]:
            return None

        records = self.species_data[species][trait]
        if len(records) < 2:
            return 0.0

        generations = [r.generation for r in records]
        values = [r.value for r in records]

        if len(set(generations)) < 2:
            return 0.0

        try:
            slope, _, r_value, _, _ = stats.linregress(generations, values)
            return slope
        except Exception as e:
            print(f"Error computing trend for {species}, trait: {trait}: {e}")
            return 0.0

    def get_evolution_summary(self, species: str) -> EvolutionSummary:
        if species not in self.species_data:
            return {"key_adaptations": [], "fitness_correlation": {}, "generations_tracked": 0}

        trait_trends = {}
        for trait in self.species_data[species]:
            trend = self.get_trait_trend(species, trait)
            if trend is not None:
                trait_trends[trait] = trend

        fitness_correlation = {}
        if species in self.fitness_data:
            fitness_by_gen = self.fitness_data[species]

            for trait in self.species_data[species]:
                trait_records = self.species_data[species][trait]

                common_gens = set(r.generation for r in trait_records) & set(fitness_by_gen.keys())

                if len(common_gens) > 1:
                    try:
                        trait_values = [next(r.value for r in trait_records if r.generation == g)
                                        for g in common_gens]
                        fitness_values = [fitness_by_gen[g] for g in common_gens]

                        if len(trait_values) > 1 and len(set(trait_values)) > 1 and len(set(fitness_values)) > 1:
                            corr, _ = stats.pearsonr(trait_values, fitness_values)
                            fitness_correlation[trait] = corr
                    except Exception as e:
                        print(f"Error while calculating correlation for {species}, trait {trait}: {e}")

        if trait_trends:
            normalized_trends = {}
            for trait, trend in trait_trends.items():
                records = self.species_data[species][trait]
                values = [r.value for r in records]
                value_range = max(values) - min(values) if len(values) > 1 else 1.0
                if value_range > 0:
                    normalized_trends[trait] = abs(trend) / value_range
                else:
                    normalized_trends[trait] = 0.0

            key_traits = sorted(normalized_trends.keys(),
                                key=lambda t: normalized_trends[t],
                                reverse=True)[:3]

            key_adaptations = [
                f"{t} {'increasing' if trait_trends[t] > 0 else 'decreasing'} ({trait_trends[t]:.4f})"
                for t in key_traits
            ]
        else:
            key_adaptations = []

        return {
            "key_adaptations": key_adaptations,
            "fitness_correlation": fitness_correlation,
            "generations_tracked": max(self.current_generation.get(species, 0), 0)
        }

    def generate_correlation_matrix(self, species: str, figsize: Tuple[int, int] = (14, 12), sns=None) -> None:
        if species not in self.species_data:
            print(f"No data available for species {species}")
            return

        traits = list(self.species_data[species].keys())

        if len(traits) < 2:
            print(f"At least 2 traits are required to generate correlations for {species}")
            return

        data = []

        generations = set()
        for trait, records in self.species_data[species].items():
            generations.update(r.generation for r in records)

        generations = sorted(generations)

        for gen in generations:
            row = {'generation': gen}

            for trait in traits:
                records = self.species_data[species][trait]

                gen_records = [r.value for r in records if r.generation == gen]
                if gen_records:
                    row[trait] = np.mean(gen_records)
                else:
                    row[trait] = np.nan

            data.append(row)

        df = pd.DataFrame(data)

        df = df.dropna()

        if len(df) < 3:
            print(
                f"Insufficient data to compute correlations (at least 3 generations with complete data required)")
            return

        corr_matrix = df.drop('generation', axis=1).corr()

        plt.figure(figsize=figsize)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .5})

        plt.title(f'Trait correlation matrix for {species}')
        plt.tight_layout()
        plt.show()

        print("\nMost significant correlations:")

        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                trait1 = corr_matrix.columns[i]
                trait2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                corr_pairs.append((trait1, trait2, corr))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        for trait1, trait2, corr in corr_pairs[:10]:
            relationship = "positive" if corr > 0 else "negative"
            strength = "very strong" if abs(corr) > 0.8 else "strong" if abs(corr) > 0.6 else "moderate"
            print(
                f"{trait1.replace('_', ' ')} and {trait2.replace('_', ' ')} have a {relationship} {strength} correlation: {corr:.3f}")

        return corr_matrix

