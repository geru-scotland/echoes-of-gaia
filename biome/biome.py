from logging import Logger

from biome.components.climate.climate import Climate
from simulation.bootstrap.context.context_data import BiomeContextData

class Biome:
    def __init__(self, context: BiomeContextData):
        self._context: BiomeContextData = context
        try:
            self._logger: Logger = self._context.logger
            # Del contexto, habrá que pasar datos de clima de los config
            self._climate = Climate()
            self._logger.info("Biome is ready!")
            self._logger.info(self._context.config.get("biome"))
        except:
            print("There was an error creating the Biome.")

