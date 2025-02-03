from typing import Any

from biome.biome import Biome

class BiomeAPI:
    def __init__(self, **kwargs: Any):
       self.biome = Biome(**kwargs)