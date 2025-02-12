from typing import Optional

import simpy

from biome.components.component import Component
from shared.enums import ComponentType


class Climate(Component):
    def __init__(self, env: simpy.Environment):
        super().__init__(ComponentType.CLIMATE, env)
        self._env.process(self._update(25))

    # Cambios drásticos de clima, deberían de ser dispatcheados mejor, esto para probar solo.
    def _update(self, delay: Optional[int] = None):
        yield self._env.timeout(delay)
        while True:
            self._logger.info(f"Updating Climate: t={self._env.now}")
            yield self._env.timeout(25)

    def get_state(self):
        pass

    def update(self):
        pass
