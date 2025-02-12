import logging

from shared.strings import Loggers


class SimulationTime:
    def __init__(self, events_per_era: int = 30000):
        self._logger: logging.Logger = logging.getLogger(Loggers.SIMULATION)
        self.events_per_era = events_per_era

    def get_current_era(self, event_count: int) -> int:
        return event_count // self.events_per_era

    def log_time(self, event_count: int):
        self._logger.info(f"Simulated events: {event_count}, Era={self.get_current_era(event_count)}")
