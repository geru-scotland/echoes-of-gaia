from abc import ABC, abstractmethod
import json

class StateHandler(ABC):
    def __init__(self):
        self._state = {}

    @abstractmethod
    def compute_state(self):
        raise NotImplementedError

    def dump_state(self) -> str:
        return json.dumps(self.compute_state(), indent=2)
