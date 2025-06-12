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
Creates timestamped directory structures for simulation experiments.

Manages output paths for evolution tracking, genetic crossover data,
and trend analysis files - it also organizes all experimental
results under unique simulation identifiers with Spanish timestamps.
Ensures proper directory hierarchy for data persistence.
"""

import os.path
from pathlib import Path
from typing import Dict, Any

from utils.paths import BASE_DIR
from datetime import datetime


class ExperimentPathManager:
    instance = None

    def __init__(self):
        self._simulation_base_path: str = None
        self._evolution_path: str = None
        self._genetic_crossover_path: str = None
        self._trends_path: str = None
        self._paths: Dict[str, Path] = {}

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = ExperimentPathManager()
        return cls.instance

    def initialize(self, sim_paths: Dict[str, Any] = None):
        timestamp = self.get_timestamp_es()
        simulation_id = f"sim_{timestamp}"
        base_path: str = os.path.join(BASE_DIR, sim_paths.get("base"), sim_paths.get("simulations"), simulation_id)

        self._paths = {
            "simulation": Path(base_path),
            "evolution": Path(os.path.join(base_path, sim_paths.get("evolution", ""))),
            "genetic_crossover": Path(os.path.join(base_path, sim_paths.get("genetic_crossover", ""))),
            "trends": Path(os.path.join(base_path, sim_paths.get("trends", ""))),
        }

        self._create_directories()

    def _create_directories(self):
        self._paths.get("simulation").mkdir(parents=True, exist_ok=True)
        self._paths.get("evolution").mkdir(parents=True, exist_ok=True)
        self._paths.get("genetic_crossover").mkdir(parents=True, exist_ok=True)
        self._paths.get("trends").mkdir(parents=True, exist_ok=True)

    def get_plot_path(self, tracker_name: str, plot_name: str) -> str:
        timestamp = self.get_timestamp_es()
        filename = f"{plot_name}_{timestamp}.png"
        return self._paths.get(tracker_name) / filename

    def get_timestamp_es(self):
        return datetime.now().strftime("%d-%B-%Y__%Hh%Mm%Ss")

    @property
    def simulation_path(self) -> str:
        return self._paths.get("simulation")

    @property
    def evolution_path(self) -> str:
        return self._paths.get("evolution")

    @property
    def genetic_crossover_path(self) -> str:
        return self._paths.get("genetic_crossover")

    @property
    def trends_path(self) -> str:
        return self._paths.get("trends")
