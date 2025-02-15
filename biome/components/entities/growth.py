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

from biome.components.component import EntityComponent
from shared.enums import ComponentType, Timers
from shared.types import ComponentData


class GrowthComponent(EntityComponent):
    def __init__(self, env: simpyEnv, growth_rate: int = 0, decay = 0):
        super().__init__(ComponentType.GROWTH, env)
        # Por ahora, ha de ser exacto y coincidir con atributos
        # en biome/data/ecosystem.json... TODO: Cambiar esto
        self._growth_rate: int = growth_rate
        self._decay: int = decay
        self._logger.warning(f"GR: {self._growth_rate}, decay: {self._decay}")
        self._env.process(self._update(Timers.Entity.GROWTH))


    def _update(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            self._logger.info(f"[Component][Growth][t={self._env.now}] Entity is growing.")
            yield self._env.timeout(timer)

    def get_state(self):
        pass
