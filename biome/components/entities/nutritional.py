"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
from typing import Optional, Callable

from simpy import Environment as simpyEnv

from biome.components.component import EntityComponent
from shared.enums import ComponentType, Timers


class NutritionalValueComponent(EntityComponent):
    def __init__(self, env: simpyEnv, callback: Callable, nutritive_value: float, nutritional_decay_rate: float,
                 toxicity: float):
        super().__init__(ComponentType.NUTRITIONAL, env, callback)
        # Por ahora, ha de ser exacto y coincidir con atributos
        # en biome/data/ecosystem.json... TODO: Cambiar esto
        self._nutritive_value = nutritive_value
        self._nutritional_decay_rate = nutritional_decay_rate
        self._toxicity = toxicity
        self._logger.debug(f"NV: {self._nutritive_value}, decay rate : {self._nutritional_decay_rate}"
                             f" toxicity: {self._toxicity}")
        self._env.process(self._update(Timers.Entity.NUTRITIONAL_VALUE_DECAY))


    def _update(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            message: str = f"[Component][{self._type}][t={self._env.now}] Entity is changing, nutrition_value: {self._nutritive_value}"
            self._handle_component_update(data=message)
            yield self._env.timeout(timer)

    def get_state(self):
        pass
