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


def main():
    parser = argparse.ArgumentParser(description="Run multiple simulations sequentially")
    parser.add_argument("--sims", type=int, default=5, help="Number of simulations to run")
    parser.add_argument("--wait", type=int, default=5, help="Wait time between simulations (seconds)")
    parser.add_argument("--config", type=str, default="training.yaml", help="Base configuration file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    exec_script_path = os.path.join(script_dir, "exec_simulation.py")

    for i in range(args.sims):
        print(f"\n{'=' * 60}")
        print(f"STARTING SIMULATION {i + 1} OF {args.sims}")
        print(f"{'=' * 60}\n")

        cmd = [sys.executable, exec_script_path]

        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, universal_newlines=True)

            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode != 0:
                print(f"Simulation {i + 1} ended with errors (code: {process.returncode})")
            else:
                print(f"Simulation {i + 1} completed successfully")

        except Exception as e:
            print(f"Error running simulation {i + 1}: {str(e)}")

        if i < args.sims - 1:
            wait_time = args.wait + random.randint(-2, 2)
            print(f"\nWaiting {wait_time} seconds before the next simulation...")
            time.sleep(wait_time)

    print("\nAll simulations have been completed.")


if __name__ == "__main__":
    main()
