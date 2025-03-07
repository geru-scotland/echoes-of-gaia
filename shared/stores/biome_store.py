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
from typing import Dict, Any

import numpy as np

from shared.enums.enums import TerrainType
from shared.types import TerrainList, BiomeStoreData
from utils.paths import BIOME_DATA_DIR

class BiomeStore:
    _initialized = False
    terrains = []
    biomes = {}
    habitats = {}
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
                cls.habitats: BiomeStoreData = data.get("habitats", {})
                cls.biomes: BiomeStoreData = data.get("biomes", {})
                cls.weather_event_effects: BiomeStoreData = data.get("weather_event_effects", {})
                cls.flora: BiomeStoreData = data.get("flora", {})
                cls.fauna: BiomeStoreData = data.get("fauna", {})
                cls.components: BiomeStoreData = data.get("components", {})

                cls._initialized = True
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading ecosystem file: {file_path}")