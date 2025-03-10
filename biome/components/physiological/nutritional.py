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
from typing import Optional

from simpy import Environment as simpyEnv

from biome.components.base.component import EntityComponent
from shared.enums.enums import ComponentType
from shared.enums.events import ComponentEvent
from shared.timers import Timers


class NutritionalValueComponent(EntityComponent):
    def __init__(self, env: simpyEnv, nutritive_value: float, nutritional_decay_rate: float,
                 toxicity: float):
        super().__init__(ComponentType.NUTRITIONAL, env)
        # Por ahora, ha de ser exacto y coincidir con atributos
        # en biome/data/ecosystem.json... TODO: Cambiar esto
        self._nutritive_value = round(nutritive_value, 2)
        self._nutritional_decay_rate = round(nutritional_decay_rate, 2)
        self._toxicity = round(toxicity, 2)
        self._logger.debug(f"NV: {self._nutritive_value}, decay rate : {self._nutritional_decay_rate}"
                             f" toxicity: {self._toxicity}")
        self._env.process(self._update(Timers.Entity.NUTRITIONAL_VALUE_DECAY))


    def _update(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            self._event_notifier.notify(ComponentEvent.UPDATE_STATE, NutritionalValueComponent, toxicity=self._toxicity)
            self._toxicity += 0.01
            yield self._env.timeout(timer)

    def get_state(self):
        pass
