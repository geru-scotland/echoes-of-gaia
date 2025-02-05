import logging

from utils.exceptions import MapGenerationError


class Map:
    def __init__(self):
        self._logger = logging.getLogger("bootstrap")
        self._logger.info("[Map] Initialising a new map.")

class MapGenerator:
    def __init__(self):
        self._logger = logging.getLogger("bootstrap")
        self._logger.info("[MapGenerator] Initialising Map generator")

    def generate(self):
        try:
            map: Map = Map()
            self._logger.info("[MapGenerator] Generating new map")
        except Exception as e:
            raise MapGenerationError(f"{e}")