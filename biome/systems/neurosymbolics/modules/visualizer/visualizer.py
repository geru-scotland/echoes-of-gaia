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

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from shared.enums.strings import Loggers


class VisualizationStrategy(ABC):

    @abstractmethod
    def visualize(self, graph: nx.DiGraph, metrics: Dict[str, Any], result: Dict[str, Any],
                  output_dir: str, timestamp: str) -> str:
        pass


class NetworkVisualization(VisualizationStrategy):

    def visualize(self, graph: nx.DiGraph, metrics: Dict[str, Any], result: Dict[str, Any],
                  output_dir: str, timestamp: str) -> str:
        plt.figure(figsize=(14, 12))
        plt.title("Estructura del Ecosistema", fontsize=18)

        env_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'environmental']
        flora_nodes = [n for n, d in graph.nodes(data=True)
                       if d.get('type') == 'species' and d.get('species_type') == 'flora']
        herbivore_nodes = [n for n, d in graph.nodes(data=True)
                           if d.get('type') == 'species' and d.get('diet') == 'herbivore']
        carnivore_nodes = [n for n, d in graph.nodes(data=True)
                           if d.get('type') == 'species' and d.get('diet') == 'carnivore']
        omnivore_nodes = [n for n, d in graph.nodes(data=True)
                          if d.get('type') == 'species' and d.get('diet') == 'omnivore']

        pos = {}
        width = 1.0

        for i, node in enumerate(env_nodes):
            angle = 2 * np.pi * i / max(len(env_nodes), 1)
            pos[node] = (width * 0.8 * np.cos(angle), 1.5 + width * 0.3 * np.sin(angle))

        for i, node in enumerate(flora_nodes):
            angle = 2 * np.pi * i / max(len(flora_nodes), 1)
            pos[node] = (width * 1.2 * np.cos(angle), width * 1.2 * np.sin(angle))

        for i, node in enumerate(herbivore_nodes):
            angle = 2 * np.pi * i / max(len(herbivore_nodes), 1)
            pos[node] = (width * 0.6 * np.cos(angle), width * 0.6 * np.sin(angle))

        for i, node in enumerate(carnivore_nodes):
            angle = 2 * np.pi * i / max(len(carnivore_nodes), 1)
            pos[node] = (width * 0.3 * np.cos(angle), width * 0.3 * np.sin(angle))

        for i, node in enumerate(omnivore_nodes):
            angle = 2 * np.pi * i / max(len(omnivore_nodes), 1)
            pos[node] = (width * 0.45 * np.cos(angle), width * 0.45 * np.sin(angle))

        node_colors = []
        node_sizes = []
        labels = {}

        for node in graph.nodes():
            node_data = graph.nodes[node]

            if node_data.get('type') == 'environmental':
                node_colors.append('skyblue')
                node_sizes.append(800)
                labels[node] = f"{node}\n({node_data.get('value', 0):.1f})"
            elif node_data.get('type') == 'species':
                if node_data.get('species_type') == 'flora':
                    node_colors.append('green')
                elif node_data.get('diet') == 'herbivore':
                    node_colors.append('yellow')
                elif node_data.get('diet') == 'carnivore':
                    node_colors.append('red')
                elif node_data.get('diet') == 'omnivore':
                    node_colors.append('orange')
                else:
                    node_colors.append('gray')

                pop = node_data.get('population', 5)
                node_sizes.append(300 + (pop * 10))
                labels[node] = f"{node}\n(pop:{pop})"

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)

        edge_colors = []
        widths = []

        for u, v, data in graph.edges(data=True):
            if data.get('type') == 'predation':
                edge_colors.append('red')
                widths.append(2.0)
            elif data.get('type') == 'resource_consumption':
                edge_colors.append('green')
                widths.append(1.5)
            elif data.get('type') == 'environmental_impact':
                edge_colors.append('blue')
                widths.append(0.8)
            else:
                edge_colors.append('gray')
                widths.append(0.5)

        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=widths,
                               arrowsize=15, alpha=0.7,
                               connectionstyle='arc3,rad=0.1')

        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=9)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Flora'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Herbívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Carnívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Omnívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Ambiental'),
            plt.Line2D([0], [0], color='red', lw=2, label='Depredación'),
            plt.Line2D([0], [0], color='green', lw=2, label='Consumo de recursos'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Impacto ambiental'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.figtext(0.02, 0.02, f"Análisis: {timestamp}", fontsize=8)

        filepath = os.path.join(output_dir, f"ecosystem_network_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath


class TrophicLevelsVisualization(VisualizationStrategy):

    def visualize(self, graph: nx.DiGraph, metrics: Dict[str, Any], result: Dict[str, Any],
                  output_dir: str, timestamp: str) -> str:
        trophic_levels = metrics.get('trophic_levels', {})

        if not trophic_levels:
            return ""

        plt.figure(figsize=(14, 10))
        plt.title("Niveles Tróficos del Ecosistema", fontsize=16)

        levels_to_species = {}
        for species, level in trophic_levels.items():
            if level not in levels_to_species:
                levels_to_species[level] = []
            levels_to_species[level].append(species)

        pos = {}
        max_level = max(levels_to_species.keys()) if levels_to_species else 0

        for level, species_list in levels_to_species.items():
            y = 1.0 - (level / (max_level + 1))
            for i, species in enumerate(species_list):
                x = (i + 1) / (len(species_list) + 1)
                pos[species] = (x, y)

        node_colors = []
        node_sizes = []

        species_nodes = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'species']

        for node in species_nodes:
            if node not in pos:
                continue

            node_data = graph.nodes[node]

            if node_data.get('species_type') == 'flora':
                node_colors.append('green')
            elif node_data.get('diet') == 'herbivore':
                node_colors.append('yellow')
            elif node_data.get('diet') == 'carnivore':
                node_colors.append('red')
            elif node_data.get('diet') == 'omnivore':
                node_colors.append('orange')
            else:
                node_colors.append('gray')

            pop = node_data.get('population', 5)
            node_sizes.append(300 + (pop * 10))

        species_subgraph = graph.subgraph(species_nodes)

        nx.draw_networkx_nodes(species_subgraph, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.8)

        predation_edges = [(u, v) for u, v, data in species_subgraph.edges(data=True)
                           if data.get('type') == 'predation']
        nx.draw_networkx_edges(species_subgraph, pos, edgelist=predation_edges,
                               edge_color='red', width=1.5,
                               arrowsize=15, alpha=0.7)

        labels = {}
        for node in species_subgraph.nodes():
            if node in trophic_levels:
                labels[node] = f"{node}"
        nx.draw_networkx_labels(species_subgraph, pos, labels=labels, font_size=9)

        for level in range(1, max_level + 1):
            y = 1.0 - (level / (max_level + 1))
            plt.axhline(y=y, color='gray', linestyle='-', alpha=0.3)
            plt.text(0.01, y + 0.01, f"Nivel {level}", fontsize=10)

            level_description = ""
            if level == 1:
                level_description = "Productores primarios"
            elif level == 2:
                level_description = "Consumidores primarios"
            elif level == 3:
                level_description = "Consumidores secundarios"
            elif level == 4:
                level_description = "Consumidores terciarios"

            if level_description:
                plt.text(0.15, y + 0.01, f"({level_description})", fontsize=9, style='italic')

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Flora'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Herbívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Carnívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Omnívoro'),
            plt.Line2D([0], [0], color='red', lw=2, label='Depredación'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.figtext(0.75, 0.15, f"Estadísticas de la red trófica:\n" +
                    f"- Niveles tróficos: {max_level}\n" +
                    f"- Especies por nivel: {', '.join([f'N{k}:{len(v)}' for k, v in levels_to_species.items()])}\n" +
                    f"- Longitud de cadena: {metrics.get('food_chain_length', 0)}",
                    bbox=dict(facecolor='white', alpha=0.8))

        filepath = os.path.join(output_dir, f"trophic_levels_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath


class CentralityVisualization(VisualizationStrategy):

    def visualize(self, graph: nx.DiGraph, metrics: Dict[str, Any], result: Dict[str, Any],
                  output_dir: str, timestamp: str) -> str:
        centrality_metrics = {}
        for metric_name in ['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality']:
            if metric_name in metrics:
                centrality_metrics[metric_name] = metrics[metric_name]

        if not centrality_metrics:
            return ""

        species_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'species']
        species_subgraph = graph.subgraph(species_nodes)

        if not species_nodes:
            return ""

        fig, axes = plt.subplots(1, len(centrality_metrics), figsize=(18, 7))
        if len(centrality_metrics) == 1:
            axes = [axes]

        fig.suptitle("Análisis de Centralidad: Especies Clave del Ecosistema", fontsize=16)

        node_colors_dict = {}
        for node in species_subgraph.nodes():
            node_data = graph.nodes[node]
            if node_data.get('species_type') == 'flora':
                node_colors_dict[node] = 'green'
            elif node_data.get('diet') == 'herbivore':
                node_colors_dict[node] = 'yellow'
            elif node_data.get('diet') == 'carnivore':
                node_colors_dict[node] = 'red'
            elif node_data.get('diet') == 'omnivore':
                node_colors_dict[node] = 'orange'
            else:
                node_colors_dict[node] = 'gray'

        try:
            pos = {}
            nodes_list = list(species_subgraph.nodes())
            for i, node in enumerate(nodes_list):
                angle = 2 * np.pi * i / len(nodes_list)
                r = 0.8
                pos[node] = (r * np.cos(angle), r * np.sin(angle))
        except:
            pos = nx.spring_layout(species_subgraph, seed=42, k=0.9)

        metric_explanations = {
            'degree_centrality': "Mide el número de conexiones directas.\nEspecies con muchas conexiones son generalistas.",
            'betweenness_centrality': "Mide la frecuencia con que una especie\nactúa como puente entre otras especies.",
            'eigenvector_centrality': "Mide la importancia global en la red.\nValor alto indica especies clave."
        }

        for i, (metric_name, centrality_values) in enumerate(centrality_metrics.items()):
            ax = axes[i]

            species_centrality = {node: value for node, value in centrality_values.items()
                                  if node in species_nodes}

            friendly_name = metric_name.replace('_', ' ').title()
            ax.set_title(friendly_name, fontsize=13)

            if metric_name in metric_explanations:
                ax.text(0.5, -0.12, metric_explanations[metric_name],
                        ha='center', va='center', transform=ax.transAxes,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

            centrality_max = max(species_centrality.values()) if species_centrality else 1

            node_sizes = {}
            node_colors = []

            for node in species_subgraph.nodes():
                centrality = species_centrality.get(node, 0)
                size_factor = (centrality / centrality_max) ** 0.5 if centrality_max > 0 else 0
                node_sizes[node] = 300 + 3000 * size_factor
                node_colors.append(node_colors_dict.get(node, 'gray'))

            nx.draw_networkx_nodes(species_subgraph, pos,
                                   node_color=node_colors,
                                   node_size=[node_sizes[n] for n in species_subgraph.nodes()],
                                   alpha=0.8, ax=ax)

            nx.draw_networkx_edges(species_subgraph, pos, alpha=0.4, ax=ax,
                                   arrows=True, arrowsize=15, width=1.2,
                                   connectionstyle='arc3,rad=0.1')

            labels = {n: n for n in species_subgraph.nodes()}
            nx.draw_networkx_labels(species_subgraph, pos, labels=labels, font_size=9, ax=ax)

            for node, (x, y) in pos.items():
                value = species_centrality.get(node, 0)
                if value > centrality_max * 0.3:
                    ax.text(x + 0.05, y + 0.05, f"{value:.2f}",
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Flora'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Herbívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Carnívoro'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Omnívoro')
        ]

        fig.legend(handles=legend_elements, loc='upper center', ncol=4,
                   bbox_to_anchor=(0.5, 0.02), frameon=True)

        keystone_species = result.get("keystone_species", [])

        if keystone_species:
            info_text = "Especies clave identificadas:\n"
            for species in keystone_species:
                species_data = graph.nodes.get(species, {})
                info_text += f"- {species} ({species_data.get('species_type', '?')})\n"

            plt.figtext(0.02, 0.02, info_text, fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))

        filepath = os.path.join(output_dir, f"centrality_analysis_{timestamp}.png")
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath


class EcosystemMetricsVisualization(VisualizationStrategy):

    def visualize(self, graph: nx.DiGraph, metrics: Dict[str, Any], result: Dict[str, Any],
                  output_dir: str, timestamp: str) -> str:
        plt.figure(figsize=(12, 10))
        plt.suptitle("Métricas y Estado del Ecosistema", fontsize=18)

        gs = plt.GridSpec(2, 2, figure=plt.gcf(), wspace=0.3, hspace=0.4)

        ax1 = plt.subplot(gs[0, 0])
        robustness = metrics.get('robustness', 0)
        connectivity = metrics.get('connectivity', 0)
        clustering = metrics.get('average_clustering', 0)

        values = [robustness, connectivity, clustering]
        labels = ['Robustez', 'Conectividad', 'Agrupamiento']

        colors = []
        for v in values:
            if v < 0.3:
                colors.append('red')
            elif v < 0.6:
                colors.append('orange')
            else:
                colors.append('green')

        ax1.bar(labels, values, color=colors)
        ax1.set_title("Robustez del Ecosistema")
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Índice (0-1)")

        for i, v in enumerate(values):
            ax1.text(i, v + 0.05, f"{v:.2f}", ha='center')

        ax2 = plt.subplot(gs[0, 1])

        trophic_structure = result.get("trophic_structure", {})
        levels_distribution = trophic_structure.get("levels_distribution", {})

        if levels_distribution:
            levels = sorted([int(k) for k in levels_distribution.keys()])
            counts = [levels_distribution[str(k)] for k in levels]

            ax2.bar(levels, counts, color='skyblue')
            ax2.set_title("Distribución de Niveles Tróficos")
            ax2.set_xlabel("Nivel Trófico")
            ax2.set_ylabel("Número de Especies")
            ax2.set_xticks(levels)

            for i, v in enumerate(counts):
                ax2.text(levels[i], v + 0.1, str(v), ha='center')
        else:
            ax2.text(0.5, 0.5, "No hay datos disponibles sobre niveles tróficos",
                     ha='center', va='center', fontsize=12)

        ax3 = plt.subplot(gs[1, 0])

        status_counts = {}
        for key, value in result.items():
            if key.endswith("_status") and key != "food_chain_status":
                if value not in status_counts:
                    status_counts[value] = 0
                status_counts[value] += 1

        if status_counts:
            status_names = list(status_counts.keys())
            status_values = list(status_counts.values())

            status_colors = []
            for status in status_names:
                if "ENDANGERED" in str(status) or "CRITICAL" in str(status):
                    status_colors.append('red')
                elif "STRESSED" in str(status) or "UNSTABLE" in str(status):
                    status_colors.append('orange')
                elif "KEYSTONE" in str(status):
                    status_colors.append('purple')
                else:
                    status_colors.append('green')

            ax3.bar(status_names, status_values, color=status_colors)
            ax3.set_title("Estado de Especies")
            ax3.set_xticklabels([str(s).split('.')[-1] for s in status_names], rotation=45, ha='right')
            ax3.set_ylabel("Número de Especies")
        else:
            ax3.text(0.5, 0.5, "No hay datos de estado disponibles",
                     ha='center', va='center', fontsize=12)

        ax4 = plt.subplot(gs[1, 1])
        ax4.axis('off')

        info_text = "DIAGNÓSTICO DEL ECOSISTEMA:\n\n"

        if "ecosystem_risk" in result:
            risk = str(result["ecosystem_risk"]).split('.')[-1].replace('_', ' ')
            info_text += f"🔴 Riesgo: {risk}\n"

        if "ecosystem_stability" in result:
            stability = str(result["ecosystem_stability"]).split('.')[-1].replace('_', ' ')
            info_text += f"🔵 Estabilidad: {stability}\n"

        if "ecosystem_structure" in result:
            structure = result["ecosystem_structure"].replace('_', ' ')
            info_text += f"🟠 Estructura: {structure}\n"

        if "niche_diversity" in result:
            diversity = result["niche_diversity"]
            info_text += f"🟢 Diversidad de nichos: {diversity}\n"

        if "recommended_action" in result:
            action = str(result["recommended_action"]).split('.')[-1].replace('_', ' ')
            info_text += f"\nRECOMENDACIÓN PRINCIPAL:\n{action}"

        ax4.text(0, 1.0, info_text, va='top', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9))

        # Guardar figura
        filepath = os.path.join(output_dir, f"ecosystem_metrics_{timestamp}.png")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath


class EcosystemVisualizer:

    def __init__(self, output_directory: str = None):
        self.strategies = {
            "network": NetworkVisualization(),
            "trophic_levels": TrophicLevelsVisualization(),
            "centrality": CentralityVisualization(),
            "metrics": EcosystemMetricsVisualization()
        }

        if output_directory is None:
            base_dir = os.path.join(os.getcwd(), "ecosystem_analysis")
        else:
            base_dir = os.path.abspath(output_directory)

        self.output_dir = base_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(Loggers.BIOME)

    def generate_visualizations(self, graph: nx.DiGraph, metrics: Dict[str, Any],
                                result: Dict[str, Any]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.output_dir, f"analysis_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        generated_paths = {}

        for name, strategy in self.strategies.items():
            try:
                filepath = strategy.visualize(graph, metrics, result, session_dir, timestamp)
                if filepath:
                    generated_paths[name] = filepath
                    self.logger.info(f"Visualización {name} generada: {filepath}")
            except Exception as e:
                self.logger.warning(f"Error generando visualización {name}: {str(e)}")

        return generated_paths
