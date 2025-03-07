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
from shared.timers import Timers


class GrowthComponent(EntityComponent):
    def __init__(self, env: simpyEnv, growth_stage: int = 0, stages: int = 3,
                 growth_rate: float = 0.05, size: float = 1.0, max_size: float = 5.0):
        super().__init__(ComponentType.GROWTH, env)
        self._growth_stage: int = growth_stage
        self._total_stages: int = stages
        self._growth_rate: float = round(growth_rate, 2)
        self._size: float = round(size, 2)
        self._max_size: float = round(max_size, 2)
        self._logger.debug(f"Stage: {self._growth_stage}/{self._total_stages}, "
                           f"Size: {self._size}/{self._max_size}, Growth rate: {self._growth_rate}")
        self._env.process(self._update_growth(Timers.Entity.GROWTH))

    def _update_growth(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            if self._growth_stage < self._total_stages and self._size < self._max_size:
                self._size += self._growth_rate
                if self._size >= self._max_size * (self._growth_stage + 1) / self._total_stages:
                    self._growth_stage += 1
                    self._notify_update(GrowthComponent, growth_stage=self._growth_stage)
                self._notify_update(GrowthComponent, size=self._size)
            yield self._env.timeout(timer)
