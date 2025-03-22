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
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Genes(ABC):

    @abstractmethod
    def convert_genes_to_components(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def validate_genes(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def lifespan(self) -> float:
        raise NotImplementedError
