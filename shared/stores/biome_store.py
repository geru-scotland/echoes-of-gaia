import json
import os
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

                cls.terrains: TerrainList = [getattr(TerrainType, t, None) for t in data.get("terrains", [])]

                cls.biomes = data.get("biomes", {})
                cls.flora = data.get("flora", {})
                cls.fauna = data.get("fauna", {})
                cls.components = data.get("components", {})

                cls._initialized = True
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading ecosystem file: {file_path}")