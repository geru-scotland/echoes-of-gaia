
import simpy

from biome.components.biome.climate import Climate
from biome.environment import Environment
from simulation.core.bootstrap.context.context_data import BiomeContextData


class Biome(Environment):
    def __init__(self, context: BiomeContextData, env: simpy.Environment):
        super().__init__(context, env)
        try:
            # Del contexto, habrá que pasar datos de clima de los config
            self._env.process(self.update(25))
            self.add_component(Climate(self._env))
            self._logger.info("Biome is ready!")
            self._logger.info(self._context.config.get("type"))
        except Exception as e:
            self._logger.info(f"There was an error creating the Biome: {e}")

    def update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            self._logger.info(f"BIOMA UPDATE!... t={self._env.now}")
            yield self._env.timeout(25)

