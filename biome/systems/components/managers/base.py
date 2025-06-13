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

"""
Base component manager class for entity component lifecycle control.

Provides generic component registration and management infrastructure;
handles active component filtering and dormancy state queries.
Supports component ID tracking and generic type parameterization for
type-safe component collections - enables modular component systems.
"""

from typing import Dict, Generic, List, Set, TypeVar

from simpy import Environment as simpyEnv

TComponent = TypeVar('TComponent')


class BaseComponentManager(Generic[TComponent]):
    def __init__(self, env: simpyEnv):
        self._env: simpyEnv = env
        self._components: Dict[int, TComponent] = {}
        self._component_ids: Set[int] = set()

    def register_component(self, component_id: int, component: TComponent) -> None:
        self._components[component_id] = component
        self._component_ids.add(component_id)

    def unregister_component(self, component_id: int) -> None:
        if component_id in self._components:
            del self._components[component_id]
            self._component_ids.discard(component_id)

    def _get_active_components(self) -> List[TComponent]:
        return [comp for comp_id, comp in self._components.items()
                if comp.is_active]

    def _get_non_dormant_components(self) -> List[TComponent]:
        return [comp for comp in self._get_active_components()
                if hasattr(comp, 'is_dormant') and not comp.is_dormant]
