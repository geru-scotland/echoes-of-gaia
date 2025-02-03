from typing import Dict, Any

from biome.systems.bootstrap.builder import BiomeBuilder
from biome.systems.context.context import Context

class Biome:
    def __init__(self, **kwargs):
        self.context: Context = BiomeBuilder(**kwargs).build()
        print("Biome Ready!")
        print(self.context)
