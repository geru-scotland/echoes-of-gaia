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
Entity descriptor dataclass for type and species identification.

Stores entity type and species information in structured format;
provides factory methods for flora and fauna creation.
Enables consistent entity identification across the biome system.
"""

from dataclasses import dataclass

from shared.enums.base import EnumBaseStr
from shared.enums.enums import EntityType, FaunaSpecies, FloraSpecies


@dataclass
class EntityDescriptor:
    entity_type: EntityType
    species: EnumBaseStr #flora o fauna

    @classmethod
    def create_flora(cls, flora_type: FloraSpecies):
        return cls(entity_type=EntityType.FLORA,species=flora_type)

    @classmethod
    def create_fauna(cls, fauna_type: FaunaSpecies):
        return cls(entity_type=EntityType.FAUNA, species=fauna_type)