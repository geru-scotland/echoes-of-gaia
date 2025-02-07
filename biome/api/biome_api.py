from biome.biome import Biome
from simulation.core.bootstrap.context.context_data import BiomeContextData


class BiomeAPI:
    def __init__(self, context: BiomeContextData):
       self.biome = Biome(context)