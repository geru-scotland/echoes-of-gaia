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
import json
import os

import numpy as np

from shared.enums import TerrainType
from shared.types import TerrainList
from utils.paths import BIOME_DATA_DIR

class BiomeStore:
    _initialized = False
    terrains = []
    biomes = {}
    flora = {}
    fauna = {}

    @classmethod
    def load_ecosystem_data(cls):
        if cls._initialized:
            return

        file_path = os.path.join(BIOME_DATA_DIR, "ecosystem.json")
        try:
            with open(file_path, "r") as file:
                data = json.load(file) or {}

                cls.terrains: TerrainList = np.array([getattr(TerrainType, t, None) for t in data.get("terrains", [])], dtype=object)

                cls.biomes = data.get("biomes", {})
                cls.flora = data.get("flora", {})
                cls.fauna = data.get("fauna", {})
                cls.components = data.get("components", {})

                cls._initialized = True
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading ecosystem file: {file_path}")