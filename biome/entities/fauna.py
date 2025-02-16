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
from typing import Any

from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums import EntityType, FaunaType


class Fauna(Entity):

    def __init__(self, env: simpyEnv, fauna_type: FaunaType):
        super().__init__(EntityType.FAUNA, env)
        self._fauna_type = fauna_type
        self._logger.error(f"FAUNA CREATED: {fauna_type}")

    def dump_components(self) -> None:
        pass

    def handle_component_update(self, **kwargs: Any):
        print(f"{self._fauna_type}({self._entity_type}) - {kwargs}")

    def compute_state(self):
        pass

    @property
    def type(self):
        return self._fauna_type