import logging
from abc import abstractmethod
from typing import Optional

from simpy import Environment as simpyEnv

from shared.enums import ComponentType
from shared.strings import Loggers


class Component:
    def __init__(self, type: ComponentType, env: simpyEnv):
        self._logger: logging.Logger = logging.getLogger(Loggers.BIOME)
        self._type: ComponentType = type
        self._env: simpyEnv = env

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def _update(self, delay: Optional[int] = None):
        pass

    @property
    def type(self):
        return self._type