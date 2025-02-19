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

from shared.enums import EntityType, FloraType


class Flora(Entity):
    def __init__(self, env: simpyEnv, flora_type: FloraType):
        super().__init__(EntityType.FLORA, env)
        self._logger.debug(f"Flora entity initialized: {flora_type}")
        self._flora_type: FloraType = flora_type

    def compute_state(self):
        pass

    def dump_components(self) -> None:
        if not self._components:
            self._logger.error(f"Entity '{self._flora_type}' has no components.")
            return

        self._logger.debug(f"Entity '{self._flora_type}' Components:")
        for component_type, component in self._components.items():
            component_attrs = vars(component)
            formatted_attrs = ", ".join(f"{k}={v.__class__}" for k, v in component_attrs.items() if not k.startswith("_"))
            self._logger.debug(f" - {component_type}: {formatted_attrs}")

    def handle_component_update(self, **kwargs: Any):
        self._logger.debug(f"{self._flora_type}({self._entity_type}) - {kwargs}")


    @property
    def type(self):
        return self._flora_type