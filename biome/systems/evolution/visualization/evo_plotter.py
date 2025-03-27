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
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats

from biome.systems.evolution.visualization.evo_tracker import EvolutionTracker
from simulation.core.experiment_path_manager import ExperimentPathManager


class EvolutionPlotter:
    def __init__(self, tracker: EvolutionTracker):
        self.tracker = tracker
        self.color_palette = plt.cm.tab10

    def plot_trait_evolution(self, species: str, traits: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        if species not in self.tracker.species_data:
            print(f"No data available for species {species}")
            return

        if traits is None:
            traits = list(self.tracker.species_data[species].keys())

        valid_traits = [t for t in traits if t in self.tracker.species_data[species]]

        if not valid_traits:
            print(f"No valid traits to visualize for {species}")
            return

        fig, axes = plt.subplots(len(valid_traits), 1, figsize=figsize, sharex=True)
        if len(valid_traits) == 1:
            axes = [axes]

        fig.suptitle(f"Trait evolution for {species}", fontsize=16)

        for i, trait in enumerate(valid_traits):
            ax = axes[i]
            records = self.tracker.species_data[species][trait]

            generations = [r.generation for r in records]
            values = [r.value for r in records]

            ax.plot(generations, values, 'o-', label=trait,
                    color=self.color_palette(i % 10))

            if len(generations) > 1:
                z = np.polyfit(generations, values, 1)
                p = np.poly1d(z)
                ax.plot(generations, p(generations), '--',
                        color=self.color_palette(i % 10), alpha=0.7,
                        label=f"Trend: {z[0]:.4f}x + {z[1]:.2f}")

            ax.set_ylabel(trait)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        axes[-1].set_xlabel("Generation")

        experiment_path_manager: ExperimentPathManager = ExperimentPathManager.get_instance()
        if experiment_path_manager:
            traits_str = "-".join(valid_traits[:min(3, len(valid_traits))])
            plot_path = experiment_path_manager.get_plot_path("evolution", f"{species}_{traits_str}_evolution")
            plt.savefig(plot_path)

            print(f"Saved trait evolution plot to: {plot_path}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

    def plot_population_trends(self, species_list: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> None:
        if not self.tracker.population_data:
            print("No population data available")
            return

        if species_list is None:
            species_list = list(self.tracker.population_data.keys())

        valid_species = [s for s in species_list if s in self.tracker.population_data]

        if not valid_species:
            print("No valid species to visualize populations")
            return

        plt.figure(figsize=figsize)

        for i, species in enumerate(valid_species):
            pop_data = self.tracker.population_data[species]
            generations = sorted(pop_data.keys())
            populations = [pop_data[g] for g in generations]

            plt.plot(generations, populations, 'o-',
                     label=species, color=self.color_palette(i % 10))

            if len(generations) > 1:
                z = np.polyfit(generations, populations, 1)
                p = np.poly1d(z)
                plt.plot(generations, p(generations), '--',
                         color=self.color_palette(i % 10), alpha=0.7)

                next_gen = max(generations) + 1
                predicted = p(next_gen)
                plt.plot(next_gen, predicted, 'x',
                         color=self.color_palette(i % 10), markersize=8)
                plt.text(next_gen, predicted, f"{predicted:.0f}",
                         verticalalignment='bottom')

        plt.title("Population trends by species")
        plt.xlabel("Generation")
        plt.ylabel("Population")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        experiment_path_manager: ExperimentPathManager = ExperimentPathManager.get_instance()
        if experiment_path_manager:
            species_str = "-".join(valid_species[:min(3, len(valid_species))])
            plot_path = experiment_path_manager.get_plot_path("trends", f"population_trends")
            plt.savefig(plot_path)
            print(f"Saved population trends plot to: {plot_path}")

        plt.tight_layout()
        plt.show()

    def plot_fitness_landscape(self, species: str, trait_x: str, trait_y: str,
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        if species not in self.tracker.species_data or \
                trait_x not in self.tracker.species_data[species] or \
                trait_y not in self.tracker.species_data[species] or \
                species not in self.tracker.fitness_data:
            print("Insufficient data to visualize fitness landscape")
            return

        trait_x_records = self.tracker.species_data[species][trait_x]
        trait_y_records = self.tracker.species_data[species][trait_y]
        fitness_by_gen = self.tracker.fitness_data[species]

        x_gens = {r.generation for r in trait_x_records}
        y_gens = {r.generation for r in trait_y_records}
        f_gens = set(fitness_by_gen.keys())
        common_gens = x_gens & y_gens & f_gens

        if len(common_gens) < 3:
            print("Insufficient data to visualize fitness landscape (at least 3 generations required)")
            return

        common_gens = sorted(common_gens)
        x_values = [next(r.value for r in trait_x_records if r.generation == g)
                    for g in common_gens]
        y_values = [next(r.value for r in trait_y_records if r.generation == g)
                    for g in common_gens]
        fitness_values = [fitness_by_gen[g] for g in common_gens]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        if len(common_gens) >= 10:
            from scipy.interpolate import griddata
            xi = np.linspace(min(x_values), max(x_values), 100)
            yi = np.linspace(min(y_values), max(y_values), 100)
            X, Y = np.meshgrid(xi, yi)
            Z = griddata((x_values, y_values), fitness_values, (X, Y), method='cubic')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        scatter = ax.scatter(x_values, y_values, fitness_values, c=common_gens,
                             cmap='plasma', s=50, alpha=1.0)

        for i in range(len(common_gens) - 1):
            ax.plot([x_values[i], x_values[i + 1]],
                    [y_values[i], y_values[i + 1]],
                    [fitness_values[i], fitness_values[i + 1]],
                    'k-', alpha=0.5)

        ax.set_xlabel(trait_x)
        ax.set_ylabel(trait_y)
        ax.set_zlabel('Fitness')
        ax.set_title(f'Fitness Landscape for {species}: {trait_x} vs {trait_y}')

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Generation')

        experiment_path_manager: ExperimentPathManager = ExperimentPathManager.get_instance()
        if experiment_path_manager:
            plot_path = experiment_path_manager.get_plot_path("evolution",
                                                              f"{species}_{trait_x}_{trait_y}_fitness_landscape")
            plt.savefig(plot_path)
            print(f"Saved fitness landscape plot to: {plot_path}")

        plt.tight_layout()
        plt.show()

    def plot_adaptive_landscape(self, species: str, figsize: Tuple[int, int] = (15, 10)) -> None:
        if species not in self.tracker.species_data:
            print(f"No data available for species {species}")
            return

        generations_data = set()
        for trait, records in self.tracker.species_data[species].items():
            generations_data.update(r.generation for r in records)

        if len(generations_data) < 2:
            print(f"Insufficient data for {species}: at least 2 distinct generations required")
            return

        traits = list(self.tracker.species_data[species].keys())
        if not traits:
            print(f"No traits recorded for {species}")
            return

        summary = self.tracker.get_evolution_summary(species)

        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)

        ax1 = fig.add_subplot(grid[0, 0:2])

        trait_completeness = {}
        for trait in traits:
            records = self.tracker.species_data[species][trait]
            trait_completeness[trait] = len(records)

        top_traits = sorted(trait_completeness.keys(),
                            key=lambda t: trait_completeness[t],
                            reverse=True)[:5]

        for i, trait in enumerate(top_traits):
            records = self.tracker.species_data[species][trait]
            generations = [r.generation for r in records]
            values = [r.value for r in records]

            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val > min_val else 1.0
            norm_values = [(v - min_val) / range_val for v in values]

            ax1.plot(generations, norm_values, 'o-',
                     label=f"{trait} [{min_val:.2f}-{max_val:.2f}]",
                     color=self.color_palette(i % 10))

            if len(generations) > 1:
                z = np.polyfit(generations, norm_values, 1)
                p = np.poly1d(z)
                ax1.plot(generations, p(generations), '--',
                         color=self.color_palette(i % 10), alpha=0.7)

        ax1.set_title("Main trait trends")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Normalized value")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left', fontsize=8)

        ax2 = fig.add_subplot(grid[0, 2])
        ax2.axis('off')

        adaptations_text = "Key adaptations:\n"
        if summary["key_adaptations"]:
            for i, adapt in enumerate(summary["key_adaptations"]):
                adaptations_text += f"{i + 1}. {adapt}\n"
        else:
            adaptations_text += "No significant adaptations detected\n"

        adaptations_text += "\nFitness correlations:\n"
        if summary["fitness_correlation"]:
            corrs = sorted(summary["fitness_correlation"].items(),
                           key=lambda x: abs(x[1]), reverse=True)
            for trait, corr in corrs[:5]:
                adaptations_text += f"{trait}: {corr:.3f}\n"
        else:
            adaptations_text += "Not enough data\n"

        adaptations_text += f"\nGenerations tracked: {summary['generations_tracked']}\n"

        if species in self.tracker.population_data:
            pop_data = self.tracker.population_data[species]
            if pop_data:
                latest_gen = max(pop_data.keys())
                adaptations_text += f"Population at gen {latest_gen}: {pop_data[latest_gen]}\n"

        ax2.text(0, 0.95, adaptations_text, verticalalignment='top',
                 fontsize=10, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        ax3 = fig.add_subplot(grid[1, 0:2])

        if species in self.tracker.population_data:
            pop_data = self.tracker.population_data[species]
            generations = sorted(pop_data.keys())
            populations = [pop_data[g] for g in generations]

            ax3.plot(generations, populations, 'o-', color='blue')

            if len(generations) > 1:
                z = np.polyfit(generations, populations, 1)
                p = np.poly1d(z)
                ax3.plot(generations, p(generations), '--', color='blue', alpha=0.7,
                         label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")

                if len(generations) >= 3:
                    next_gen = max(generations) + 1
                    predicted = p(next_gen)
                    ax3.plot(next_gen, predicted, 'x', color='red', markersize=8)
                    ax3.text(next_gen, predicted, f"Pred: {predicted:.0f}",
                             verticalalignment='bottom')

        ax3.set_title("Population over time")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Population")
        ax3.grid(True, linestyle='--', alpha=0.7)
        if len(generations) > 1:
            ax3.legend()

        ax4 = fig.add_subplot(grid[1, 2])

        if species in self.tracker.fitness_data:
            fitness_data = self.tracker.fitness_data[species]
            generations = sorted(fitness_data.keys())
            fitness_values = [fitness_data[g] for g in generations]

            ax4.plot(generations, fitness_values, 'o-', color='green')

            if len(generations) > 1:
                z = np.polyfit(generations, fitness_values, 1)
                p = np.poly1d(z)
                ax4.plot(generations, p(generations), '--', color='green', alpha=0.7,
                         label=f"Trend: {z[0]:.2f}x + {z[1]:.2f}")

        ax4.set_title("Fitness over time")
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Average fitness")
        ax4.grid(True, linestyle='--', alpha=0.7)
        if len(generations) > 1:
            ax4.legend()

        if len(top_traits) >= 2 and summary["generations_tracked"] >= 3:
            ax5 = fig.add_subplot(grid[2, :])

            trait1, trait2 = top_traits[0], top_traits[1]

            records1 = self.tracker.species_data[species][trait1]
            records2 = self.tracker.species_data[species][trait2]

            gens1 = {r.generation for r in records1}
            gens2 = {r.generation for r in records2}
            common_gens = sorted(gens1 & gens2)

            if common_gens:
                values1 = [next(r.value for r in records1 if r.generation == g)
                           for g in common_gens]
                values2 = [next(r.value for r in records2 if r.generation == g)
                           for g in common_gens]

                scatter = ax5.scatter(values1, values2, c=common_gens,
                                      cmap='viridis', s=100, alpha=0.7)

                for i in range(len(common_gens) - 1):
                    ax5.annotate("",
                                 xy=(values1[i + 1], values2[i + 1]),
                                 xytext=(values1[i], values2[i]),
                                 arrowprops=dict(arrowstyle="->", lw=1.5, color='red'))

                    ax5.annotate(f"Gen {common_gens[i]}",
                                 xy=(values1[i], values2[i]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8)

                ax5.annotate(f"Gen {common_gens[-1]}",
                             xy=(values1[-1], values2[-1]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=8)

                cbar = plt.colorbar(scatter, ax=ax5)
                cbar.set_label('Generation')

                ax5.set_title(f"Coevolution of {trait1} and {trait2}")
                ax5.set_xlabel(trait1)
                ax5.set_ylabel(trait2)
                ax5.grid(True, linestyle='--', alpha=0.7)

                if len(common_gens) > 2:
                    corr, p_val = stats.pearsonr(values1, values2)
                    ax5.text(0.05, 0.95, f"Correlation: {corr:.3f} (p={p_val:.3f})",
                             transform=ax5.transAxes, fontsize=10,
                             verticalalignment='top',
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.suptitle(f"Evolutionary Overview of {species}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        experiment_path_manager: ExperimentPathManager = ExperimentPathManager.get_instance()
        if experiment_path_manager:
            plot_path = experiment_path_manager.get_plot_path("evolution", f"{species}_adaptive_landscape")
            plt.savefig(plot_path)
            print(f"Saved adaptive landscape plot to: {plot_path}")

        plt.show()
