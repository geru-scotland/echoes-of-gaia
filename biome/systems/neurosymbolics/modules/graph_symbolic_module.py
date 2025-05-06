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
from typing import Dict, Any

import networkx as nx

from biome.systems.neurosymbolics.modules.visualizer.visualizer import EcosystemVisualizer
from shared.enums.enums import RecommendedAction, SpeciesStatus, InterventionPriority, SpeciesAction, StabilityStatus, \
    BiodiversityStatus, PredatorPreyBalance, EcosystemRisk
from shared.enums.strings import Loggers
from shared.types import SymbolicFeedback, Observation
from utils.loggers import LoggerManager


class GraphBasedSymbolicModule:
    def __init__(self, rules_config: Dict[str, Any] = None):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self.rules = rules_config or {}
        self._ecosystem_graph = nx.DiGraph()
        self._logger.info(f"Graph-based symbolic module ready")

        self._ecosystem_graph = nx.DiGraph()
        self._visualizer = EcosystemVisualizer("graphs")

    def infer(self, observation: Observation) -> SymbolicFeedback:
        result = {"context": {}}

        species_data = observation.get("species_data", {})
        biome_data = observation.get("biome_data", {})

        self._build_ecosystem_graph(species_data, biome_data)

        graph_metrics = self._analyze_graph()
        result["graph_metrics"] = graph_metrics

        self._analyze_species_status(species_data, result)
        self._analyze_predator_prey_dynamics(biome_data, result)
        self._analyze_ecosystem_stability(biome_data, result)

        self._analyze_species_centrality(result)
        self._analyze_trophic_levels(result)
        self._analyze_ecosystem_robustness(result)
        try:
            visualization_paths = self._visualizer.generate_visualizations(
                self._ecosystem_graph, graph_metrics, result)

            result["visualizations"] = visualization_paths

            self._logger.info(f"Visualizaciones generadas: {len(visualization_paths)}")
            self._logger.info(f"Informe HTML: {visualization_paths.get('summary', 'No generado')}")
        except Exception as e:
            self._logger.error(f"Error generando visualizaciones: {str(e)}")
        return result

    def _build_ecosystem_graph(self, species_data: Dict[str, Dict[str, Any]], biome_data: Dict[str, Any]) -> None:
        self._ecosystem_graph.clear()

        for env_factor in ['temperature', 'humidity', 'precipitation']:
            if env_factor in biome_data:
                self._ecosystem_graph.add_node(env_factor,
                                               type='environmental',
                                               value=biome_data.get(env_factor, 0))

        for species_name, data in species_data.items():
            self._ecosystem_graph.add_node(species_name,
                                           type='species',
                                           species_type=data.get('type', 'unknown'),
                                           diet=data.get('diet', 'unknown'),
                                           population=data.get('population', 0),
                                           biomass=data.get('biomass', 0),
                                           stress=data.get('avg_stress', 0))

        for species_name, data in species_data.items():
            species_type = data.get('type')
            diet_type = data.get('diet')

            for env_factor in ['temperature', 'humidity', 'precipitation']:
                if env_factor in self._ecosystem_graph:
                    self._ecosystem_graph.add_edge(env_factor, species_name,
                                                   type='environmental_impact',
                                                   weight=1.0)

            if species_type == 'flora':
                self._ecosystem_graph.add_edge('humidity', species_name,
                                               type='resource_consumption',
                                               weight=0.8)
                self._ecosystem_graph.add_edge('precipitation', species_name,
                                               type='resource_consumption',
                                               weight=0.9)

            elif species_type == 'fauna':
                if diet_type == 'herbivore':
                    for other_species, other_data in species_data.items():
                        if other_data.get('type') == 'flora':
                            self._ecosystem_graph.add_edge(species_name, other_species,
                                                           type='predation',
                                                           weight=0.7)

                elif diet_type in ['carnivore', 'omnivore']:
                    for other_species, other_data in species_data.items():
                        if (other_data.get('type') == 'fauna' and
                                other_data.get('diet') == 'herbivore'):
                            self._ecosystem_graph.add_edge(species_name, other_species,
                                                           type='predation',
                                                           weight=0.9)

                    if diet_type == 'omnivore':
                        for other_species, other_data in species_data.items():
                            if other_data.get('type') == 'flora':
                                self._ecosystem_graph.add_edge(species_name, other_species,
                                                               type='predation',
                                                               weight=0.5)

    def _analyze_graph(self) -> Dict[str, Any]:
        metrics = {}

        if not self._ecosystem_graph.nodes():
            return metrics

        try:
            metrics['degree_centrality'] = nx.degree_centrality(self._ecosystem_graph)
            metrics['betweenness_centrality'] = nx.betweenness_centrality(self._ecosystem_graph)
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(self._ecosystem_graph, max_iter=1000)
        except:
            self._logger.warning("Error calculating centrality metrics")

        species_nodes = [n for n, d in self._ecosystem_graph.nodes(data=True)
                         if d.get('type') == 'species']
        species_subgraph = self._ecosystem_graph.subgraph(species_nodes)

        metrics['connectivity'] = nx.density(species_subgraph)
        metrics['average_clustering'] = nx.average_clustering(self._ecosystem_graph.to_undirected())

        try:
            communities = list(nx.community.greedy_modularity_communities(
                self._ecosystem_graph.to_undirected()))
            metrics['community_count'] = len(communities)
            metrics['community_sizes'] = [len(c) for c in communities]
        except:
            self._logger.warning("Error detecting communities")

        try:
            robustness = self._estimate_robustness()
            metrics['robustness'] = robustness
        except:
            self._logger.warning("Error calculating robustness")

        try:
            trophic_levels = self._calculate_trophic_levels()
            metrics['trophic_levels'] = trophic_levels
            metrics['food_chain_length'] = max(trophic_levels.values()) if trophic_levels else 0
        except:
            self._logger.warning("Error calculating trophic levels")

        return metrics

    def _estimate_robustness(self) -> float:
        species_nodes = [n for n, d in self._ecosystem_graph.nodes(data=True)
                         if d.get('type') == 'species']

        if not species_nodes:
            return 0.0

        original_connectivity = nx.density(self._ecosystem_graph.subgraph(species_nodes))

        import random
        num_to_remove = max(1, int(len(species_nodes) * 0.2))
        nodes_to_remove = random.sample(species_nodes, num_to_remove)

        test_graph = self._ecosystem_graph.copy()
        test_graph.remove_nodes_from(nodes_to_remove)

        remaining_species = [n for n, d in test_graph.nodes(data=True)
                             if d.get('type') == 'species']

        if not remaining_species:
            return 0.0

        reduced_connectivity = nx.density(test_graph.subgraph(remaining_species))

        # Robustez = cuánto se mantiene la conectividad tras eliminar nodos
        # Un valor cercano a 1 indica alta robustez
        if original_connectivity > 0:
            return reduced_connectivity / original_connectivity
        return 0.0

    def _calculate_trophic_levels(self) -> Dict[str, int]:
        trophic_levels = {}

        for node, data in self._ecosystem_graph.nodes(data=True):
            if data.get('type') == 'species' and data.get('species_type') == 'flora':
                trophic_levels[node] = 1

        fauna_nodes = [n for n, d in self._ecosystem_graph.nodes(data=True)
                       if d.get('type') == 'species' and d.get('species_type') == 'fauna']

        for _ in range(len(fauna_nodes)):
            for node in fauna_nodes:
                prey_nodes = list(self._ecosystem_graph.successors(node))
                prey_levels = [trophic_levels.get(prey, 0) for prey in prey_nodes
                               if prey in trophic_levels]

                if prey_levels:
                    trophic_levels[node] = 1 + max(prey_levels)

        return trophic_levels

    def _analyze_species_status(self, species_data, result):
        for species, data in species_data.items():
            if data.get("population", 0) < 5:
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

        if predator_prey_ratio > 0.5:
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

    def _analyze_species_centrality(self, result):
        graph_metrics = result.get("graph_metrics", {})
        centrality_metrics = graph_metrics.get("eigenvector_centrality", {})

        if not centrality_metrics:
            return

        species_centrality = {node: value for node, value in centrality_metrics.items()
                              if self._ecosystem_graph.nodes[node].get('type') == 'species'}

        if species_centrality:
            max_centrality = max(species_centrality.values())
            threshold = max_centrality * 0.7

            keystone_species = [species for species, value in species_centrality.items()
                                if value >= threshold]

            if keystone_species:
                result["keystone_species"] = keystone_species
                result["keystone_protection_priority"] = InterventionPriority.HIGH

                for species in keystone_species:
                    species_type = self._ecosystem_graph.nodes[species].get('species_type')
                    if species_type == 'fauna':
                        result[f"{species}_status"] = SpeciesStatus.KEYSTONE
                        result[f"{species}_action"] = SpeciesAction.PROTECTION_NEEDED

    def _analyze_trophic_levels(self, result):
        graph_metrics = result.get("graph_metrics", {})
        trophic_levels = graph_metrics.get("trophic_levels", {})

        if not trophic_levels:
            return

        levels_to_species = {}
        for species, level in trophic_levels.items():
            if level not in levels_to_species:
                levels_to_species[level] = []
            levels_to_species[level].append(species)

        result["trophic_structure"] = {
            "levels_count": len(levels_to_species),
            "levels_distribution": {str(k): len(v) for k, v in levels_to_species.items()}
        }

        food_chain_length = graph_metrics.get("food_chain_length", 0)
        if food_chain_length < 2:
            result["food_chain_status"] = "underdeveloped"
            result["recommended_action"] = RecommendedAction.INTRODUCE_SPECIES_DIVERSITY
        elif food_chain_length > 4:
            result["food_chain_status"] = "complex"
            if len(levels_to_species.get(food_chain_length, [])) < 2:
                result["ecosystem_risk"] = EcosystemRisk.TOP_PREDATOR_VULNERABILITY

    def _analyze_ecosystem_robustness(self, result):
        graph_metrics = result.get("graph_metrics", {})
        robustness = graph_metrics.get("robustness", 0)

        if robustness < 0.4:
            result["ecosystem_stability"] = StabilityStatus.VULNERABLE
            result["recommended_action"] = RecommendedAction.INCREASE_BIODIVERSITY
        elif robustness > 0.7:
            result["ecosystem_stability"] = StabilityStatus.RESILIENT

        connectivity = graph_metrics.get("connectivity", 0)
        if connectivity < 0.2:
            result["ecosystem_structure"] = "fragmented"
            result["ecosystem_risk"] = EcosystemRisk.FRAGMENTATION
        elif connectivity > 0.7:
            result["ecosystem_structure"] = "highly_connected"
            result["ecosystem_risk"] = EcosystemRisk.CASCADE_FAILURE

        community_count = graph_metrics.get("community_count", 0)

        if community_count <= 1:
            result["niche_diversity"] = "low"
            result["recommended_action"] = RecommendedAction.INTRODUCE_SPECIES_DIVERSITY
        elif community_count >= 3:
            result["niche_diversity"] = "high"

    def update_rules(self, new_rules: Dict[str, Any]) -> None:
        self.rules.update(new_rules)
