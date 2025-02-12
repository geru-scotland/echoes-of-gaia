import logging

from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums import EntityType
from shared.strings import Loggers


class Flora(Entity):
    def __init__(self, env: simpyEnv):
        super().__init__(EntityType.FLORA, env)
        self._logger.info("Flora entity initialized")