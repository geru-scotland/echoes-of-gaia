from simulation.bootstrap.builders.biome_builder import BiomeBuilder
from simulation.bootstrap.context.context import Context
from utils.loggers import setup_logger


class Bootstrap:
    def __init__(self):
        self._logger = setup_logger("bootstrap", "bootstrap.log")
        self._context: Context = Context(self._logger)
        self._build_context()

    def _build_context(self):
        try:
            biome_builder: BiomeBuilder = BiomeBuilder(self._logger)
            self._context.set("biome_ctx", biome_builder.build())
            # TODO: Repetir para el sim context
        except Exception as e:
            self._logger.exception(f"There was an error building the context: {e}")

    def get_context(self) -> Context:
        return self._context
