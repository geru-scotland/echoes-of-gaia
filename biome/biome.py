from logging import Logger

from simulation.bootstrap.context.context_data import BiomeContextData

class Biome:
    def __init__(self, context: BiomeContextData):
        self._context: BiomeContextData = context
        try:
            self._logger: Logger = self._context.logger
            self._logger.info("Biome is ready!")
        except:
            print("There was an error creating the Biome.")

