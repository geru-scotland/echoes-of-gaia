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

import os
import sys
import time
import argparse
import subprocess
import random


def print_simulation_header(sim_number):
    header = f" SIMULATION {sim_number} ".center(60, "=")
    print(f"\n{header}\n")


def main():
    parser = argparse.ArgumentParser(description="Run multiple simulations sequentially")
    parser.add_argument("--sims", type=int, default=300, help="Number of simulations to run")
    parser.add_argument("--wait", type=int, default=5, help="Wait time between simulations (seconds)")
    parser.add_argument("--config", type=str, default="training.yaml", help="Base configuration file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exec_script_path = os.path.join(script_dir, "exec_simulation.py")

    for i in range(args.sims):
        print_simulation_header(i + 1)

        cmd = [sys.executable, exec_script_path]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, universal_newlines=True)

            for line in process.stdout:
                print(line, end='')

            process.wait()

        except Exception as e:
            print(f"Error running simulation {i + 1}: {str(e)}")

        if i < args.sims - 1:
            wait_time = args.wait
            print(f"\nWaiting {wait_time} seconds before the next simulation...\n")
            time.sleep(wait_time)

    print("\n" + "=" * 60)
    print(" ALL SIMULATIONS HAVE BEEN COMPLETED ".center(60, "="))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
