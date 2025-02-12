from logging import Logger

import simpy

from biome.components.climate.climate import Climate
from simulation.core.bootstrap.context.context_data import BiomeContextData

class Biome:
    def __init__(self, context: BiomeContextData, env: simpy.Environment):
        self._context: BiomeContextData = context
        try:
            self._env = env
            self._logger: Logger = self._context.logger
            # Del contexto, habrá que pasar datos de clima de los config
            self._env.process(self.update(25))
            self._climate = Climate(self._env)
            self._logger.info("Biome is ready!")
            self._logger.info(self._context.config.get("type"))
        except Exception as e:
            print(f"There was an error creating the Biome: {e}")

    def update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            print(f"BIOMA UPDATE!... t={self._env.now}")
            yield self._env.timeout(25)

