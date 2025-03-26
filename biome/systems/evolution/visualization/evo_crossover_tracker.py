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
from typing import Dict, List, Tuple, Optional, Union, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from biome.systems.evolution.genes.fauna_genes import FaunaGenes
from biome.systems.evolution.genes.flora_genes import FloraGenes
from biome.systems.evolution.visualization.evo_plotter import EvolutionPlotter
from shared.enums.events import SimulationEvent
from shared.enums.strings import Loggers
from simulation.core.systems.events.event_bus import SimulationEventBus
from utils.loggers import LoggerManager


class GeneticCrossoverTracker:
    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.EVOLUTION_AGENT)
        self.crossover_events = []
        self.gene_ranges = {}
        self.species_events = {}
        SimulationEventBus.register(SimulationEvent.SIMULATION_FINISHED, self._handle_simulation_finished)

    def _handle_simulation_finished(self, *args, **kwargs):
        self._logger.info("Simulation finished, generating genetic evolution visualizations...")
        for species in self.species_events.keys():
            self._logger.info(f"Generating genetic summary for {species}...")
            generate_genetic_evolution_summary(self, species)

    def register_crossover(self,
                           species: str,
                           generation: int,
                           parent_genes: List[Union[FloraGenes, FaunaGenes]],
                           offspring_gene: Union[FloraGenes, FaunaGenes]) -> int:
        parent_traits = []
        for parent in parent_genes:
            traits = {}
            for trait_name in dir(parent):
                if not trait_name.startswith('_') and not callable(getattr(parent, trait_name)):
                    value = getattr(parent, trait_name)
                    if isinstance(value, (int, float)):
                        traits[trait_name] = value
                        if trait_name not in self.gene_ranges:
                            self.gene_ranges[trait_name] = (value, value)
                        else:
                            min_val, max_val = self.gene_ranges[trait_name]
                            self.gene_ranges[trait_name] = (min(min_val, value), max(max_val, value))
            parent_traits.append(traits)

        offspring_traits = {}
        for trait_name in dir(offspring_gene):
            if not trait_name.startswith('_') and not callable(getattr(offspring_gene, trait_name)):
                value = getattr(offspring_gene, trait_name)
                if isinstance(value, (int, float)):
                    offspring_traits[trait_name] = value
                    if trait_name not in self.gene_ranges:
                        self.gene_ranges[trait_name] = (value, value)
                    else:
                        min_val, max_val = self.gene_ranges[trait_name]
                        self.gene_ranges[trait_name] = (min(min_val, value), max(max_val, value))

        event = {
            'species': species,
            'generation': generation,
            'parents': parent_traits,
            'offspring': offspring_traits
        }

        self.crossover_events.append(event)
        event_index = len(self.crossover_events) - 1

        if species not in self.species_events:
            self.species_events[species] = []
        self.species_events[species].append(event_index)

        return event_index

    def analyze_mutation_impact(self, event_index: int) -> Dict[str, float]:
        if event_index >= len(self.crossover_events):
            return {}

        event = self.crossover_events[event_index]
        parents = event['parents']
        offspring = event['offspring']

        mutation_impact = {}

        for trait, value in offspring.items():
            parent_values = [p.get(trait) for p in parents if trait in p]

            if parent_values:
                avg_parent = sum(v for v in parent_values if v is not None) / len(parent_values)
                if trait in self.gene_ranges:
                    min_val, max_val = self.gene_ranges[trait]
                    trait_range = max_val - min_val
                    if trait_range > 0:
                        relative_impact = (value - avg_parent) / trait_range
                        mutation_impact[trait] = relative_impact
                    else:
                        mutation_impact[trait] = 0.0
                else:
                    mutation_impact[trait] = 0.0
            else:
                mutation_impact[trait] = 1.0

        return mutation_impact

    def visualize_crossover(self, event_index: int, figsize: Tuple[int, int] = (12, 10)) -> None:
        if event_index >= len(self.crossover_events):
            print(f"Event not found: {event_index}")
            return

        event = self.crossover_events[event_index]
        parents = event['parents']
        offspring = event['offspring']

        all_traits = set()
        for p in parents:
            all_traits.update(p.keys())
        all_traits.update(offspring.keys())

        growth_traits = [t for t in all_traits if 'growth' in t.lower()]
        resist_traits = [t for t in all_traits if 'resist' in t.lower()]
        photo_traits = [t for t in all_traits if 'photo' in t.lower() or 'respiration' in t.lower()]
        nutrition_traits = [t for t in all_traits if 'nutri' in t.lower() or 'toxic' in t.lower()]
        temperature_traits = [t for t in all_traits if
                              'temperature' in t.lower() or 'heat' in t.lower() or 'cold' in t.lower()]
        other_traits = [t for t in all_traits if t not in growth_traits + resist_traits +
                        photo_traits + nutrition_traits + temperature_traits]

        trait_groups = []
        for group in [growth_traits, resist_traits, photo_traits, nutrition_traits, temperature_traits, other_traits]:
            if group:
                trait_groups.append(sorted(group))

        traits = [t for group in trait_groups for t in group]

        fig, axes = plt.subplots(2, 1, figsize=figsize,
                                 gridspec_kw={'height_ratios': [1, 1.5], 'hspace': 0.3})

        ax_genes = axes[0]

        num_parents = len(parents)
        num_traits = len(traits)
        data = np.zeros((num_parents + 1, num_traits))

        for i, parent in enumerate(parents):
            for j, trait in enumerate(traits):
                if trait in parent:
                    min_val, max_val = self.gene_ranges.get(trait, (0, 1))
                    value_range = max_val - min_val
                    if value_range > 0:
                        data[i, j] = (parent[trait] - min_val) / value_range
                    else:
                        data[i, j] = 0.5
                else:
                    data[i, j] = np.nan

        for j, trait in enumerate(traits):
            if trait in offspring:
                min_val, max_val = self.gene_ranges.get(trait, (0, 1))
                value_range = max_val - min_val
                if value_range > 0:
                    data[num_parents, j] = (offspring[trait] - min_val) / value_range
                else:
                    data[num_parents, j] = 0.5
            else:
                data[num_parents, j] = np.nan

        cmap = LinearSegmentedColormap.from_list("gene_cmap", ["#4575b4", "#91bfdb", "#ffffbf", "#fc8d59", "#d73027"])

        sns.heatmap(data, annot=False, cmap=cmap, ax=ax_genes,
                    cbar_kws={'label': 'Normalized value'})

        ax_genes.set_yticks(np.arange(0.5, num_parents + 1))
        ax_genes.set_yticklabels(['Parent ' + str(i + 1) for i in range(num_parents)] + ['Offspring'])

        ax_genes.set_xticks(np.arange(0.5, num_traits))
        ax_genes.set_xticklabels([t.replace('_', '\n') for t in traits], rotation=45, ha='right')

        ax_genes.set_title(f"Genetic crossover in {event['species']} - Generation {event['generation']}")
        ax_mutation = axes[1]

        mutation_impact = self.analyze_mutation_impact(event_index)

        impact_values = []
        for trait in traits:
            impact_values.append(mutation_impact.get(trait, 0.0))

        bars = ax_mutation.barh(traits, impact_values, height=0.7)

        for i, bar in enumerate(bars):
            impact = impact_values[i]
            if impact > 0.1:
                bar.set_color('#d73027')
            elif impact > 0.02:
                bar.set_color('#fc8d59')
            elif impact > -0.02:
                bar.set_color('#ffffbf')
            elif impact > -0.1:
                bar.set_color('#91bfdb')
            else:
                bar.set_color('#4575b4')

        ax_mutation.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        max_abs_impact = max(abs(min(impact_values)), abs(max(impact_values)))
        ax_mutation.set_xlim(-max_abs_impact * 1.1, max_abs_impact * 1.1)

        ax_mutation.set_xlabel('Mutation impact')
        ax_mutation.set_ylabel('Traits')
        ax_mutation.set_title('Mutation impact analysis per trait')

        ax_mutation.set_yticklabels([t.replace('_', ' ') for t in traits])

        for i, trait in enumerate(traits):
            offspring_val = offspring.get(trait, None)
            if offspring_val is not None:
                parent_vals = [p.get(trait, None) for p in parents]
                parent_vals = [v for v in parent_vals if v is not None]

                if parent_vals:
                    avg_parent = sum(parent_vals) / len(parent_vals)
                    change = offspring_val - avg_parent
                    percent_change = (change / avg_parent) * 100 if avg_parent != 0 else float('inf')

                    annotation = f"{offspring_val:.3f} ({percent_change:+.1f}%)"

                    x_pos = impact_values[i]
                    if x_pos >= 0:
                        x_pos = x_pos + max_abs_impact * 0.05
                        ax_mutation.text(x_pos, i, annotation, va='center', fontsize=8)
                    else:
                        x_pos = x_pos - max_abs_impact * 0.05
                        ax_mutation.text(x_pos, i, annotation, va='center', ha='right', fontsize=8)

        plt.tight_layout()
        plt.show()
    def visualize_gene_evolution(self, species: str, traits: List[str],
                                 generations: Optional[List[int]] = None,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        if species not in self.species_events:
            print(f"No data for species {species}")
            return

        events = [self.crossover_events[i] for i in self.species_events[species]]

        if generations:
            events = [e for e in events if e['generation'] in generations]

        if not events:
            print("Not enough data to visualize")
            return

        generation_data = {}
        for event in events:
            gen = event['generation']
            if gen not in generation_data:
                generation_data[gen] = []
            generation_data[gen].append(event['offspring'])

        gen_averages = {}
        gen_stddevs = {}

        for gen, offsprings in generation_data.items():
            trait_values = {trait: [] for trait in traits}

            for offspring in offsprings:
                for trait in traits:
                    if trait in offspring:
                        trait_values[trait].append(offspring[trait])

            gen_averages[gen] = {trait: np.mean(values) if values else np.nan
                                 for trait, values in trait_values.items()}

            gen_stddevs[gen] = {trait: np.std(values) if len(values) > 1 else 0.0
                                for trait, values in trait_values.items()}

        generations = sorted(generation_data.keys())

        if len(generations) < 2:
            print("At least two generations are required to visualize evolution")
            return

        num_traits = len(traits)
        fig, axes = plt.subplots(num_traits, 1, figsize=figsize, sharex=True)

        if num_traits == 1:
            axes = [axes]

        for i, trait in enumerate(traits):
            ax = axes[i]

            values = [gen_averages[gen].get(trait, np.nan) for gen in generations]
            errors = [gen_stddevs[gen].get(trait, 0.0) for gen in generations]

            ax.errorbar(generations, values, yerr=errors, marker='o', markersize=8,
                        linewidth=2, elinewidth=1, capsize=5)

            valid_indices = ~np.isnan(values)
            if np.sum(valid_indices) > 2:
                x_valid = np.array(generations)[valid_indices]
                y_valid = np.array(values)[valid_indices]

                z = np.polyfit(x_valid, y_valid, 2)
                p = np.poly1d(z)

                x_smooth = np.linspace(min(generations), max(generations), 100)
                y_smooth = p(x_smooth)

                ax.plot(x_smooth, y_smooth, '--', alpha=0.6)

                if z[0] < 0.001 and z[0] > -0.001:
                    eq_text = f"y = {z[1]:.4f}x + {z[2]:.4f}"
                else:
                    eq_text = f"y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}"

                ax.text(0.02, 0.92, eq_text, transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

            ax.set_ylabel(trait.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.6)

            if trait in self.gene_ranges:
                min_val, max_val = self.gene_ranges[trait]
                range_buffer = (max_val - min_val) * 0.1
                ax.set_ylim(min_val - range_buffer, max_val + range_buffer)

                if len(values) > 0 and not np.isnan(values[0]):
                    ax.axhline(y=values[0], color='gray', linestyle='--', alpha=0.5)
                    ax.text(generations[-1], values[0], "Initial value",
                            verticalalignment='bottom', horizontalalignment='right',
                            color='gray', fontsize=8)

        axes[-1].set_xlabel('Generation')

        plt.suptitle(f'Gene evolution for {species}', fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()
    def visualize_evolutionary_path(self, species: str, traits: List[str],
                                    figsize: Tuple[int, int] = (12, 10)) -> None:
        if len(traits) != 2:
            print("Exactly two traits are required for this visualization")
            return

        if species not in self.species_events:
            print(f"No data for species {species}")
            return

        events = [self.crossover_events[i] for i in self.species_events[species]]

        if not events:
            print("Not enough data to visualize")
            return

        generation_data = {}
        for event in events:
            gen = event['generation']
            if gen not in generation_data:
                generation_data[gen] = []
            generation_data[gen].append(event['offspring'])

        gen_averages = {}

        for gen, offsprings in generation_data.items():
            trait_values = {trait: [] for trait in traits}

            for offspring in offsprings:
                for trait in traits:
                    if trait in offspring:
                        trait_values[trait].append(offspring[trait])

            gen_averages[gen] = {trait: np.mean(values) if values else np.nan
                                 for trait, values in trait_values.items()}

        generations = sorted(generation_data.keys())

        if len(generations) < 2:
            print("At least two generations are required to visualize evolution")
            return

        trait1, trait2 = traits
        x_values = [gen_averages[gen].get(trait1, np.nan) for gen in generations]
        y_values = [gen_averages[gen].get(trait2, np.nan) for gen in generations]

        valid_indices = ~(np.isnan(x_values) | np.isnan(y_values))
        if np.sum(valid_indices) < 2:
            print("Not enough valid data for both traits")
            return

        generations = np.array(generations)[valid_indices]
        x_values = np.array(x_values)[valid_indices]
        y_values = np.array(y_values)[valid_indices]

        fig, ax = plt.subplots(figsize=figsize)

        scatter = ax.scatter(x_values, y_values, c=generations,
                             cmap='plasma', s=100, alpha=0.8)

        for i in range(len(generations) - 1):
            ax.arrow(x_values[i], y_values[i],
                     x_values[i + 1] - x_values[i],
                     y_values[i + 1] - y_values[i],
                     head_width=0.02 * (max(y_values) - min(y_values)),
                     head_length=0.02 * (max(x_values) - min(x_values)),
                     fc='red', ec='red', alpha=0.6)

            ax.annotate(f"Gen {generations[i]}",
                        (x_values[i], y_values[i]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=9)

        ax.annotate(f"Gen {generations[-1]}",
                    (x_values[-1], y_values[-1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9)

        cbar = plt.colorbar(scatter)
        cbar.set_label('Generation')

        trait1_name = trait1.replace('_', ' ').title()
        trait2_name = trait2.replace('_', ' ').title()

        ax.set_xlabel(trait1_name)
        ax.set_ylabel(trait2_name)

        plt.title(f'Evolutionary path for {species}: {trait1_name} vs {trait2_name}')

        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
def generate_genetic_evolution_summary(tracker: GeneticCrossoverTracker, species: str) -> None:
    if species not in tracker.species_events:
        print(f"No evolutionary data for species {species}")
        return

    event_indices = tracker.species_events[species]
    events = [tracker.crossover_events[i] for i in event_indices]

    generations = sorted(set(e['generation'] for e in events))

    if len(generations) < 2:
        print(f"Not enough generations to analyze species {species}")
        return

    all_traits = set()
    for event in events:
        all_traits.update(event['offspring'].keys())

    trait_variations = {}
    for trait in all_traits:
        values_by_gen = {}
        for event in events:
            gen = event['generation']
            if trait in event['offspring']:
                if gen not in values_by_gen:
                    values_by_gen[gen] = []
                values_by_gen[gen].append(event['offspring'][trait])

        if len(values_by_gen) >= 2:
            gen_averages = {g: np.mean(vals) for g, vals in values_by_gen.items()}

            gens = sorted(gen_averages.keys())
            if len(gens) >= 2:
                first_gen = gens[0]
                last_gen = gens[-1]

                if gen_averages[first_gen] != 0:
                    relative_change = (gen_averages[last_gen] - gen_averages[first_gen]) / abs(gen_averages[first_gen])
                    trait_variations[trait] = abs(relative_change)

    top_traits = sorted(trait_variations.keys(), key=lambda t: trait_variations[t], reverse=True)[:5]

    print(f"\n=== Genetic Evolution Summary for {species} ===\n")
    print(f"Tracked generations: {generations}")
    print(f"Registered crossover events: {len(events)}")
    print(f"Traits with highest variation: {[t.replace('_', ' ').title() for t in top_traits]}")

    print("\nGenerating visualizations...\n")

    last_event_index = event_indices[-1]
    print(f"Visualizing last crossover event (gen {tracker.crossover_events[last_event_index]['generation']})")
    tracker.visualize_crossover(last_event_index)

    print(f"Visualizing main trait evolution")
    tracker.visualize_gene_evolution(species, top_traits)

    if len(top_traits) >= 2:
        print(f"Visualizing evolutionary path for the top two traits")
        tracker.visualize_evolutionary_path(species, top_traits[:2])

    print("\n===== End of Summary =====\n")
