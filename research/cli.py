"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
import os
import sys
import argparse
from enum import Enum, auto
from threading import Thread
from typing import Dict, List, Callable, Optional, TypedDict, Any, Tuple, Union
import subprocess

from config.settings import Settings
from shared.strings import Loggers
from simulation.api.simulation_api import SimulationAPI
from simulation.render.manager import RenderManager
from utils.loggers import LoggerManager, LogColors


class MenuOptionType(Enum):
    SIMULATION = auto()
    TRAINING = auto()
    VISUALIZATION = auto()
    SETTINGS = auto()
    QUIT = auto()


class MenuOptionConfig(TypedDict):
    description: str
    handler: Callable[[], None]
    color: str


class CLIMenuManager:
    def __init__(self, settings: Settings):
        self._settings: Settings = settings
        self._logger = LoggerManager.get_logger(Loggers.RESEARCH)
        self._running: bool = True

        self._menu_options: Dict[MenuOptionType, MenuOptionConfig] = {
            MenuOptionType.SIMULATION: {
                "description": "Run Biome Simulation",
                "handler": self._run_simulation,
                "color": LogColors.GREEN
            },
            MenuOptionType.TRAINING: {
                "description": "Train Models",
                "handler": self._run_training,
                "color": LogColors.BRIGHT_BLUE
            },
            MenuOptionType.VISUALIZATION: {
                "description": "Visualize Results",
                "handler": self._run_visualization,
                "color": LogColors.BRIGHT_YELLOW
            },
            MenuOptionType.SETTINGS: {
                "description": "Settings",
                "handler": self._manage_settings,
                "color": LogColors.BRIGHT_CYAN
            },
            MenuOptionType.QUIT: {
                "description": "Quit",
                "handler": self._quit,
                "color": LogColors.BRIGHT_RED
            }
        }

    def _print_header(self) -> None:
        header = f"""
{LogColors.GREEN}
    ███████╗ ██████╗██╗  ██╗ ██████╗ ███████╗███████╗     ██████╗ ███████╗     ██████╗  █████╗ ██╗ █████╗ 
    ██╔════╝██╔════╝██║  ██║██╔═══██╗██╔════╝██╔════╝    ██╔═══██╗██╔════╝    ██╔════╝ ██╔══██╗██║██╔══██╗
    █████╗  ██║     ███████║██║   ██║█████╗  ███████╗    ██║   ██║█████╗      ██║  ███╗███████║██║███████║
    ██╔══╝  ██║     ██╔══██║██║   ██║██╔══╝  ╚════██║    ██║   ██║██╔══╝      ██║   ██║██╔══██║██║██╔══██║
    ███████╗╚██████╗██║  ██║╚██████╔╝███████╗███████║    ╚██████╔╝██║         ╚██████╔╝██║  ██║██║██║  ██║
    ╚══════╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝     ╚═════╝ ╚═╝          ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{LogColors.BRIGHT_WHITE}
                              Research Hub - A simulation framework for intelligent Biomas 
{LogColors.RESET}
        """
        print(header)

    def _display_menu(self) -> None:
        self._clear_screen()
        self._print_header()

        print(f"\n{LogColors.BRIGHT_WHITE}Select an option:{LogColors.RESET}\n")

        for idx, (option_type, option_config) in enumerate(self._menu_options.items(), 1):
            description = option_config["description"]
            color = option_config["color"]
            print(f"  {color}[{idx}] {description}{LogColors.RESET}")

        print()

    def _get_user_choice(self) -> Optional[MenuOptionType]:
        try:
            sys.stdout.write(
                f"{LogColors.BRIGHT_WHITE}Enter your choice (1-{len(self._menu_options)}): {LogColors.RESET}")
            sys.stdout.flush()
            choice = input()
            idx = int(choice) - 1

            if 0 <= idx < len(self._menu_options):
                return list(self._menu_options.keys())[idx]
            else:
                print(
                    f"\n{LogColors.BRIGHT_RED}Invalid choice. Please select a number between 1 and {len(self._menu_options)}.{LogColors.RESET}")
                return None
        except ValueError:
            print(f"\n{LogColors.BRIGHT_RED}Please enter a valid number.{LogColors.RESET}")
            return None

    def _clear_screen(self) -> None:
        os.system('cls' if os.name == 'nt' else 'clear')

    def _confirm_action(self, message: str) -> bool:
        sys.stdout.write(f"{LogColors.YELLOW}{message} (y/n): {LogColors.RESET}")
        sys.stdout.flush()
        response = input().lower()
        return response in ['y', 'yes']

    def _run_simulation(self) -> None:
        self._clear_screen()
        print(f"\n{LogColors.GREEN}=== Biome Simulation ==={LogColors.RESET}\n")

        print(f"{LogColors.BRIGHT_BLUE}Initializing simulation...{LogColors.RESET}")
        self._logger.info("Starting simulation script")

        simulation_script = os.path.join("simulation", "main.py")

        try:
            if os.path.exists(simulation_script):
                print(f"{LogColors.BRIGHT_YELLOW}Running simulation...{LogColors.RESET}")
                print(f"{LogColors.GRAY}Press Ctrl+C to abort{LogColors.RESET}")
                subprocess.run([sys.executable, simulation_script], check=True)
            else:
                print(
                    f"{LogColors.BRIGHT_RED}Error: Simulation script not found at {simulation_script}{LogColors.RESET}")
                print(f"{LogColors.GRAY}Make sure the project structure is correct.{LogColors.RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{LogColors.BRIGHT_RED}Error running simulation script: {e}{LogColors.RESET}")
        except Exception as e:
            print(f"{LogColors.BRIGHT_RED}Unexpected error: {e}{LogColors.RESET}")

        sys.stdout.write(
            f"\n{LogColors.GREEN}Simulation complete. Press Enter to return to the main menu...{LogColors.RESET}")
        sys.stdout.flush()
        input()

    def _run_training(self) -> None:
        self._clear_screen()
        print(f"\n{LogColors.BRIGHT_BLUE}=== Training Models ==={LogColors.RESET}\n")

        training_options = {
            1: ("Fauna Agent (RL)", self._train_fauna_agent),
            2: ("Climate Model", self._train_climate_model),
            3: ("Return to Main Menu", lambda: None)
        }

        for idx, (description, _) in training_options.items():
            print(f"  {LogColors.BRIGHT_BLUE}[{idx}] {description}{LogColors.RESET}")

        try:
            sys.stdout.write(f"\n{LogColors.BRIGHT_WHITE}Select training option: {LogColors.RESET}")
            sys.stdout.flush()
            choice = int(input())
            if choice in training_options:
                if choice < 3:
                    training_options[choice][1]()
            else:
                print(f"\n{LogColors.BRIGHT_RED}Invalid choice.{LogColors.RESET}")
                sys.stdout.write(f"{LogColors.GRAY}Press Enter to continue...{LogColors.RESET}")
                sys.stdout.flush()
                input()
        except ValueError:
            print(f"\n{LogColors.BRIGHT_RED}Please enter a valid number.{LogColors.RESET}")
            sys.stdout.write(f"{LogColors.GRAY}Press Enter to continue...{LogColors.RESET}")
            sys.stdout.flush()
            input()

    def _train_fauna_agent(self) -> None:
        print(f"\n{LogColors.YELLOW}Fauna Agent Training (RL) - Not yet implemented{LogColors.RESET}")
        print(
            f"{LogColors.GRAY}This feature will allow training reinforcement learning agents for fauna behavior{LogColors.RESET}")
        sys.stdout.write(f"\n{LogColors.GRAY}Press Enter to return...{LogColors.RESET}")
        sys.stdout.flush()
        input()

    def _train_climate_model(self) -> None:
        print(f"\n{LogColors.YELLOW}Climate Model Training - Not yet implemented{LogColors.RESET}")
        print(
            f"{LogColors.GRAY}This feature will allow training climate prediction and variation models{LogColors.RESET}")
        input(f"\n{LogColors.GRAY}Press Enter to return...{LogColors.RESET}")

    def _run_visualization(self) -> None:
        self._clear_screen()
        print(f"\n{LogColors.BRIGHT_YELLOW}=== Visualization Tools ==={LogColors.RESET}\n")
        print(f"{LogColors.YELLOW}Launching Snapshot Viewer...{LogColors.RESET}")

        try:
            visualization_script = os.path.join("simulation", "visualization", "main.py")
            if os.path.exists(visualization_script):
                print(f"{LogColors.BRIGHT_BLUE}Starting visualization viewer...{LogColors.RESET}")
                subprocess.run([sys.executable, visualization_script], check=True)
            else:
                print(
                    f"{LogColors.BRIGHT_RED}Error: Visualization script not found at {visualization_script}{LogColors.RESET}")
                print(f"{LogColors.GRAY}Make sure the project structure is correct.{LogColors.RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{LogColors.BRIGHT_RED}Error running visualization tool: {e}{LogColors.RESET}")
        except Exception as e:
            print(f"{LogColors.BRIGHT_RED}Unexpected error: {e}{LogColors.RESET}")

        sys.stdout.write(f"\n{LogColors.GRAY}Press Enter to return to the main menu...{LogColors.RESET}")
        sys.stdout.flush()
        input()

    def _manage_settings(self) -> None:
        self._clear_screen()
        print(f"\n{LogColors.BRIGHT_CYAN}=== Settings ==={LogColors.RESET}\n")
        print(f"{LogColors.CYAN}Settings management is not yet implemented.{LogColors.RESET}")
        print(
            f"{LogColors.GRAY}This feature will allow configuring simulation and training parameters.{LogColors.RESET}")
        input(f"\n{LogColors.GRAY}Press Enter to return to the main menu...{LogColors.RESET}")

    def _quit(self) -> None:
        if self._confirm_action("Are you sure you want to quit?"):
            self._running = False
            print(f"\n{LogColors.BRIGHT_GREEN}Thank you for using Echoes of Gaia!{LogColors.RESET}")
            self._logger.info("Application terminated by user")

    def run(self) -> None:
        while self._running:
            self._display_menu()
            choice = self._get_user_choice()

            if choice:
                self._menu_options[choice]["handler"]()

                if choice != MenuOptionType.QUIT and self._running:
                    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Echoes of Gaia Research Hub")

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visualization)"
    )

    return parser.parse_args()


def main() -> None:
    if "TERM" not in os.environ:
        os.environ["TERM"] = "xterm-256color"

    args = parse_args()

    settings = Settings()
    LoggerManager.initialize(args.log_level)

    cli_manager = CLIMenuManager(settings)

    try:
        cli_manager.run()
    except KeyboardInterrupt:
        print(f"\n\n{LogColors.BRIGHT_RED}Program interrupted by user.{LogColors.RESET}")
        LoggerManager.get_logger(Loggers.RESEARCH).info("Application terminated by Ctrl+C")
    except Exception as e:
        print(f"\n{LogColors.BRIGHT_RED}An unexpected error occurred: {e}{LogColors.RESET}")
        LoggerManager.get_logger(Loggers.RESEARCH).exception("Unexpected error")

    print(f"\n{LogColors.RESET}Exiting Echoes of Gaia. Goodbye!{LogColors.RESET}")


if __name__ == "__main__":
    main()