from typing import Optional

from simpy import Environment as simpyEnv

from biome.components.component import EntityComponent
from shared.enums import ComponentType, Timers


class GrowthComponent(EntityComponent):
    def __init__(self, env: simpyEnv, rate: str):
        super().__init__(ComponentType.GROWTH, env)
        self._rate = rate
        self._env.process(self._update(Timers.Entity.GROWTH))

    def _update(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            self._logger.info(f"[Component][Growth][t={self._env.now}] Entity is growing.")
            yield self._env.timeout(timer)

    def get_state(self):
        pass
