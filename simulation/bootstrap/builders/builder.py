from abc import ABC
from logging import Logger
from typing import Dict, Any, Optional

from config.settings import Settings
from simulation.bootstrap.context.context import Context


class ConfiguratorStrategy(ABC):
    def configure(self, settings: Settings, **kwargs: Any) -> None:
        raise NotImplementedError


class Builder(ABC):
    def __init__(self, logger: Logger):
        self._logger = logger
        self._context: Optional[Context] = None

    def _initialise(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    @property
    def context(self) -> Context:
        return self._context


