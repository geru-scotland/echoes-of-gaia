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
import argparse
import time
from typing import Dict, Any

from config.settings import Settings
from research.training.reinforcement.config.training_config_manager import TrainingConfigManager
from shared.enums.enums import SimulationMode, DietType
from simulation.api.simulation_api import SimulationAPI
from utils.paths import DATASET_GENERATED_CONFIGS_DIR


def check_fauna_extinction(simulation) -> bool:
    biome = simulation.get_biome()
    entity_provider = biome.get_entity_provider()

    fauna = entity_provider.get_fauna(only_alive=True)

    if len(fauna) == 0:
        return True

    remaining_species = set()
    for entity in fauna:
        remaining_species.add(entity.get_species())

    if len(remaining_species) == 1:
        sample_entity = fauna[0]

        if hasattr(sample_entity, 'diet_type') and sample_entity.diet_type == DietType.CARNIVORE:
            return True

    return False


def run_simulation_with_extinction_check(config_file: str, config: Dict[str, Any],
                                         extinction_check_interval: int = 100) -> None:
    settings = Settings(override_configs=config_file)
    simulation = SimulationAPI(settings=settings, mode=SimulationMode.NORMAL)

    simulation.initialise_training()

    total_steps = config['simulation']['eras']['events-per-era'] * config.get('simulation', {}).get('eras', {}).get(
        'amount', 10)
    current_step = 0

    while current_step < total_steps:
        steps_to_run = min(extinction_check_interval, total_steps - current_step)
        simulation.step(steps_to_run)
        current_step += steps_to_run

        if check_fauna_extinction(simulation):
            print(f"Extinction detected in the step {current_step}. Finishing simulation...")
            break

    simulation.finish_training()


def main():
    parser = argparse.ArgumentParser(description="Generates simulation data for LSTM training")
    parser.add_argument("--base-config", type=str, default="training.yaml",
                        help="Name of the base configuration file")
    parser.add_argument("--sims", type=int, default=1,
                        help="Number of simulations to run")
    parser.add_argument("--check-interval", type=int, default=100,
                        help="Interval to check for extinction")
    parser.add_argument("--output-dir", type=str, default="generated_configs",
                        help="Directory to save configurations")
    args = parser.parse_args()

    for i in range(args.sims):

        random_config = TrainingConfigManager.generate_random_config(args.base_config)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        biome_type = random_config['biome']['type']

        filename = f"config_{biome_type}_{timestamp}.yaml"
        temp_config_path = TrainingConfigManager.save_temp_config(random_config, filename,
                                                                  DATASET_GENERATED_CONFIGS_DIR)
        # config = enhance_configuration_variability(config)

        print(f"Configuration saved at: {temp_config_path}")

        try:
            run_simulation_with_extinction_check(temp_config_path, random_config, args.check_interval)
        except Exception as e:
            print(f"Error in simulation {i + 1}: {str(e)}")

        if i < args.sims - 1:
            print("Waiting 5 seconds before the next simulation...")
            time.sleep(5)


if __name__ == "__main__":
    main()
