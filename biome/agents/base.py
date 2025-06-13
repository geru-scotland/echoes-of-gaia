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
Abstract base class for agent implementation with generic types.

Defines standard agent interface with perceive-decide-act cycle;
supports type-safe state and action handling through generics.
Enables consistent agent behavior patterns across the biome system.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TState = TypeVar("TState")
TAction = TypeVar("TAction")

class Agent(Generic[TState, TAction], ABC):

    @abstractmethod
    def perceive(self) -> TState:
        raise NotImplementedError

    @abstractmethod
    def decide(self, observation: TState) -> TAction:
        raise NotImplementedError

    @abstractmethod
    def act(self, action: TAction) -> None:
        raise NotImplementedError
