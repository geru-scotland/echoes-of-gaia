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

from shared.enums.enums import EntityType, FaunaSpecies
from shared.types import Target


class TrainingTargetManager:
    _target: Target = tuple()
    _acquired: bool = False
    _current_episode: int = 0
    training_mode: bool = False

    @classmethod
    def set_target(cls, species: FaunaSpecies, entity_type: EntityType, generation: int) -> None:
        cls._target = (species, entity_type, generation)
        cls._acquired = False

    @classmethod
    def mark_as_done(cls) -> None:
        cls._acquired = True

    @classmethod
    def is_valid_target(cls, species: FaunaSpecies, entity_type: EntityType, generation: int) -> bool:
        return cls._target == (species, entity_type, generation)

    @classmethod
    def is_acquired(cls) -> bool:
        return cls._acquired

    @classmethod
    def reset(cls) -> None:
        cls._target = tuple()
        cls._acquired = False

    @classmethod
    def get_current_episode(cls) -> int:
        return cls._current_episode

    @classmethod
    def set_episode(cls, episode: int) -> None:
        cls._current_episode = episode

    @classmethod
    def get_target(cls) -> Optional[Target]:
        return cls._target if cls._target else None

    @classmethod
    def set_training_mode(cls, is_training: bool) -> None:
        cls._training_mode = is_training

    @classmethod
    def is_training_mode(cls) -> bool:
        return cls._training_mode

    @classmethod
    def mark_as_acquired(cls) -> None:
        cls._acquired = True
