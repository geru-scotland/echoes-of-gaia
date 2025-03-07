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

from biome.entities.descriptor import EntityDescriptor
from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums.enums import FaunaSpecies
from shared.types import HabitatList


class Fauna(Entity):

    def __init__(self, id: int, env: simpyEnv, fauna_type: FaunaSpecies, habitats: HabitatList):
        descriptor: EntityDescriptor = EntityDescriptor.create_fauna(fauna_type)
        super().__init__(id, env, descriptor, habitats)
        self._fauna_type = fauna_type
        self._logger.debug(f"FAUNA CREATED: {fauna_type}")
        self._habitats: HabitatList = habitats

    def _register_events(self):
        pass

    def dump_components(self) -> None:
        pass

    def compute_state(self):
        pass
