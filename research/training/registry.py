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
Environment registry for managing reinforcement learning environments.

Provides a centralized mechanism for registering and accessing training
environments by agent type or class name. Implements a decorator-based
registration system for dynamic environment discovery - ensures correct
environment selection for specific agent types during training.
"""

from typing import Dict, Callable, Type

from typing_extensions import TypeVar

from shared.enums.enums import Agents

T = TypeVar("T")


class EnvironmentRegistry:
    _environments: Dict[Agents.Reinforcement, Type] = {}
    _class_names: Dict[str, Type] = {}

    @classmethod
    def register(cls, agent_type: Agents.Reinforcement) -> Callable[[Type[T]], Type[T]]:
        def wrapper(environment_class: Type[T]) -> Type[T]:
            cls._environments[agent_type] = environment_class
            cls._class_names[environment_class.__name__] = environment_class
            return environment_class

        return wrapper

    @classmethod
    def get(cls, agent_type: Agents.Reinforcement) -> Type:
        return cls._environments.get(agent_type, None)

    @classmethod
    def get_by_name(cls, class_name: str) -> Type:
        return cls._class_names.get(class_name, None)
