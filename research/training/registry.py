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
from typing import Dict, Callable, Type

from typing_extensions import TypeVar

from shared.enums.enums import Agents


T = TypeVar("T")

class EnvironmentRegistry:
    _environments: Dict[Agents.Reinforcement, Callable] = {}

    @classmethod
    def register(cls, agent_type: Agents.Reinforcement) -> Type[T]:
        def wrapper(environment_class_name: Type[T]):
            cls._environments[agent_type] = environment_class_name
            return environment_class_name
        return wrapper

    @classmethod
    def get(cls, agent_type: Agents.Reinforcement) -> Type[T]:
        return cls._environments.get(agent_type, None)