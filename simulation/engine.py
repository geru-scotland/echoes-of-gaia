from biome.api.biome_api import BiomeAPI


class SimulationEngine:
    def __init__(self):
        self.biome_api = BiomeAPI()

    def run(self):
        print("[SimulationEngine] Running simulation...")



simulation = SimulationEngine()
simulation.run()