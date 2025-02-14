from typing import Optional

from simpy import Environment as simpyEnv

from biome.components.component import EntityComponent
from shared.enums import ComponentType, Timers


class NutritionalValueComponent(EntityComponent):
    def __init__(self, env: simpyEnv, nutritive_value: float, nutritional_decay_rate: float,
                 toxicity: float):
        super().__init__(ComponentType.NUTRITIONAL, env)
        # Por ahora, ha de ser exacto y coincidir con atributos
        # en biome/data/ecosystem.json... TODO: Cambiar esto
        self._nutritive_value = nutritive_value
        self._nutritional_decay_rate = nutritional_decay_rate
        self._toxicity = toxicity
        self._logger.warning(f"NV: {self._nutritive_value}, decay rate : {self._nutritional_decay_rate}"
                             f" toxicity: {self._toxicity}")
        self._env.process(self._update(Timers.Entity.NUTRITIONAL_VALUE_DECAY))


    def _update(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            self._logger.info(f"[Component][NUTRITIONAL][t={self._env.now}] Nutritional value is decaying.")
            yield self._env.timeout(timer)

    def get_state(self):
        pass
