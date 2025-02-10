from abc import ABC, abstractmethod


class EventHandler(ABC):
    def __init__(self):
        self._register_events()

    @abstractmethod
    def _register_events(self):
        raise NotImplementedError


