""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""
from typing import Optional

from simpy import Environment as simpyEnv
from biome.components.component import EntityComponent
from shared.enums import ComponentType, Timers


class VitalComponent(EntityComponent):
    def __init__(self, env: simpyEnv, health: float = 100.0, max_health: float = 100.0,
                 age: float = 0.0, aging_rate: float = 0.01, health_decay_rate: float = 0.005):
        super().__init__(ComponentType.VITAL, env)
        self._health: float = round(health, 2)
        self._max_health: float = round(max_health, 2)
        self._age: float = round(age, 2)
        self._aging_rate: float = round(aging_rate, 2)
        self._health_decay_rate: float = round(health_decay_rate, 2)
        self._alive: bool = True
        self._logger.debug(f"Health: {self._health}/{self._max_health}, Age: {self._age}, "
                           f"Aging rate: {self._aging_rate}, Health decay: {self._health_decay_rate}")
        self._env.process(self._update_age(Timers.Entity.AGING))
        self._env.process(self._update_health(Timers.Entity.HEALTH_DECAY))

    def _update_age(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            # TODO: el decay rate ha de ser escalado con _age
            # pensar bien en formulación / valores.
            self._age += self._aging_rate
            self._notify_update(age=self._age)
            yield self._env.timeout(timer)

    def _update_health(self, timer: Optional[int] = None):
        yield self._env.timeout(timer)
        while True:
            if self._health > 0:
                self._health -= self._health_decay_rate
                if self._health <= 0:
                    self._health = 0
                    self._alive = False
                    self._notify_update(alive=self._alive)
                self._notify_update(health=self._health)
            yield self._env.timeout(timer)

    # TODO: Métodos con soporte para recibir daño/heals