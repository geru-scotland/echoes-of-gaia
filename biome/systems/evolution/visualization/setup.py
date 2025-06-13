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
Evolution visualization system setup and entity registration utilities.

Configures evolution tracker with plotter integration for trait analysis;
provides registration functions for evolved entities and populations.
Handles correlation matrix generation and visualization coordination;
supports fitness tracking and generation-based trait monitoring.
"""

from biome.systems.evolution.genetics import extract_genes_from_entity
from biome.systems.evolution.visualization.evo_plotter import EvolutionPlotter
from biome.systems.evolution.visualization.evo_tracker import EvolutionTracker
from shared.enums.events import SimulationEvent
from simulation.core.systems.events.event_bus import SimulationEventBus


def setup_evolution_visualization_system():
    evolution_tracker = EvolutionTracker()

    def handle_simulation_finished(*args, **kwargs):
        print("Generating evolutionary visualizations...")

        plotter = EvolutionPlotter(evolution_tracker)

        print("\n=== REGISTERED TRAITS BY SPECIES ===")
        for species, traits_data in evolution_tracker.species_data.items():
            print(f"\nSpecies: {species}")
            print(f"Available traits: {list(traits_data.keys())}")

            flora_specific = [
                "base_photosynthesis_efficiency",
                "base_respiration_rate",
                "nutrient_absorption_rate",
                "mycorrhizal_rate",
                "base_toxicity"
            ]

            available_flora_traits = [t for t in flora_specific if t in traits_data]
            print(f"Available flora traits: {available_flora_traits}")

            if available_flora_traits:
                try:
                    print(f"Generating visualization for specific traits of {species}...")
                    plotter.plot_trait_evolution(species, available_flora_traits,
                                                 figsize=(12, 4 * len(available_flora_traits)))
                except Exception as e:
                    print(f"Error while visualizing specific traits: {e}")
        plotter = EvolutionPlotter(evolution_tracker)

        if not evolution_tracker.species_data:
            print("No evolutionary data was recorded during the simulation.")
            return

        for species in evolution_tracker.species_data.keys():
            print(f"Generating visualization for {species}...")
            plotter.plot_adaptive_landscape(species)

        if evolution_tracker.population_data:
            plotter.plot_population_trends()

        print("\nGenerating correlation matrices between traits...")
        for species in evolution_tracker.species_data.keys():
            try:
                print(f"\nCorrelations for {species}:")
                evolution_tracker.generate_correlation_matrix(species)
            except Exception as e:
                print(f"Error while generating correlation matrix for {species}: {e}")

    SimulationEventBus.register(SimulationEvent.SIMULATION_FINISHED, handle_simulation_finished)

    return evolution_tracker


def register_evolved_entity(tracker: EvolutionTracker, entity, generation: int, fitness: float = None):
    try:
        species = str(entity.get_species())
        genes = extract_genes_from_entity(entity)
        trait_values = {}

        for trait_name in [
            "growth_modifier",
            "growth_efficiency",
            "max_size",
            "max_vitality",
            "aging_rate",
            "health_modifier",
            "cold_resistance",
            "heat_resistance",
            "optimal_temperature"
        ]:
            if hasattr(genes, trait_name):
                trait_values[trait_name] = getattr(genes, trait_name)

        if hasattr(genes, "base_photosynthesis_efficiency"):
            flora_traits = [
                "base_photosynthesis_efficiency",
                "base_respiration_rate",
                "metabolic_activity",
                "nutrient_absorption_rate",
                "mycorrhizal_rate",
                "base_nutritive_value",
                "base_toxicity"
            ]

            for trait_name in flora_traits:
                if hasattr(genes, trait_name):
                    trait_values[trait_name] = getattr(genes, trait_name)

        tracker.register_trait_values(species, generation, trait_values)

        if fitness is not None:
            tracker.register_fitness(species, generation, fitness)

        return True

    except Exception as e:
        print(f"Error while registering evolved entity: {e}")
        return False


def update_species_population(tracker: EvolutionTracker, species: str, generation: int, count: int):
    tracker.register_population(species, generation, count)


def register_generation_fitness(tracker: EvolutionTracker, species: str, generation: int, fitness: float):
    tracker.register_fitness(species, generation, fitness)


def initialize_evolution_tracking():
    evolution_tracker = setup_evolution_visualization_system()
    return evolution_tracker
