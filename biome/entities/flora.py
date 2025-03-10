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

from biome.entities.descriptor import EntityDescriptor
from biome.entities.entity import Entity
from simpy import Environment as simpyEnv

from shared.enums.enums import FloraSpecies
from shared.enums.events import ComponentEvent
from shared.types import HabitatList


class Flora(Entity):
    def __init__(self, id: int, env: simpyEnv, flora_type: FloraSpecies, habitats: HabitatList):
        descriptor: EntityDescriptor = EntityDescriptor.create_flora(flora_type)
        super().__init__(id, env, descriptor, habitats)
        self._logger.debug(f"Flora entity initialized: {flora_type}")
        self._flora_type: FloraSpecies = flora_type
        self._is_dormant: bool = False

    def _register_events(self):
        super()._register_events()
        self._event_notifier.register(ComponentEvent.DORMANCY_TOGGLE, self._handle_toggle_dormancy)

    def _handle_toggle_dormancy(self, *args, **kwargs) -> None:
        dormant: bool = kwargs.get("dormant", False)
        if self._is_dormant != dormant:
            self._is_dormant = dormant
            # Sincronizo a todos los componentes.
            self._event_notifier.notify(ComponentEvent.DORMANCY_UPDATED)

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
