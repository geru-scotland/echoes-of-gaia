from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums import EntityType


class Fauna(Entity):
    def __init__(self, env: simpyEnv):
        super().__init__(EntityType.FAUNA, env)