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
# shared/evolution/ranges.py
from typing import Dict, Tuple, Union

Number = Union[int, float]

FLORA_GENE_RANGES: Dict[str, Tuple[Number, Number]] = {
    "growth_modifier": (0.1, 2.0),
    "growth_efficiency": (0.3, 1.0),
    "max_size": (0.1, 5.0),
    "max_vitality": (50.0, 200.0),
    "aging_rate": (0.1, 2.0),
    "health_modifier": (0.4, 2.0),
    "base_photosynthesis_efficiency": (0.4, 1.0),
    "base_respiration_rate": (0.05, 1.0),
    "lifespan": (1.0, 1000.0),
    "metabolic_activity": (0.2, 1.0),
    "max_energy_reserves": (50.0, 150.0),
    "cold_resistance": (0.0, 1.0),
    "heat_resistance": (0.0, 1.0),
    "optimal_temperature": (-30, 50),
    "nutrient_absorption_rate": (0.1, 1.0),
    "mycorrhizal_rate": (0.01, 0.04),
    "base_nutritional_value": (0.1, 1.0),
    "base_toxicity": (0.01, 1.0)
}