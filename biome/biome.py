from logging import Logger
from typing import Dict, Any, Optional

from biome.systems.bootstrap.builder import BiomeBuilder
from biome.systems.context.context import Context

class Biome:
    def __init__(self, **kwargs: Dict[str, Any]):
        self._context: Optional[Context] = None
        self._logger: Optional[Logger] = None
        self._bootstrap(**kwargs)

    def _bootstrap(self, **kwargs: Dict[str, Any]):
        self._context, self._logger = BiomeBuilder(**kwargs).build()
        self._logger.info("Biome Ready!")
        self._logger.info(self._context)


