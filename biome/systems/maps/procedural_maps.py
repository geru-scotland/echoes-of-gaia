import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List

from shared.constants import MAP_DEFAULT_SIZE
from utils.exceptions import MapGenerationError

@dataclass
class Map:
    size: Tuple[int, int] = MAP_DEFAULT_SIZE,
    weights: List[int] = field(default_factory=list)

    def __post_init__(self):
        logger = logging.getLogger("bootstrap")
        logger.info(f"[Map] Initialised with size={self.size} and weights={self.weights}")

class MapGenerator:
    def __init__(self):
        self._logger = logging.getLogger("bootstrap")
        self._logger.info("[MapGenerator] Initialising Map generator")

    def generate(self, map_data: Dict[str, Any]) -> Map:
        try:
            self._logger.info("[MapGenerator] Generating new map...")
            return Map(**map_data)
        except Exception as e:
            raise MapGenerationError(f"{e}")