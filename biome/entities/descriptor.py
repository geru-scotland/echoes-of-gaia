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
from dataclasses import dataclass

from shared.base import EnumBaseStr
from shared.enums import EntityType, FloraType, FaunaType


@dataclass
class EntityDescriptor:
    entity_type: EntityType
    specific_type: EnumBaseStr #flora o fauna

    @classmethod
    def create_flora(cls, flora_type: FloraType):
        return cls(entity_type=EntityType.FLORA, specific_type=flora_type)

    @classmethod
    def create_fauna(cls, fauna_type: FaunaType):
        return cls(entity_type=EntityType.FAUNA, specific_type=fauna_type)