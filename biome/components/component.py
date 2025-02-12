from abc import abstractmethod
from typing import Optional

from simpy import Environment as simpyEnv

from shared.enums import ComponentType


class Component:
    def __init__(self, env: simpyEnv, type: ComponentType):
        self._component_type: ComponentType = type
        self._env: simpyEnv = env

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def _update(self, delay: Optional[int] = None):
        pass

    @property
    def type(self):
        return self._component_type