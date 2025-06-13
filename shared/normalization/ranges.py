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
from typing import Dict, Tuple, Union

Number = Union[int, float]

CLIMATE_RANGES: Dict[str, Tuple[Number, Number]] = {
    "temperature": (-30.0, 51.0),        # °C
    "humidity": (2.0, 100.0),            # %
    "precipitation": (0.0, 1000.0),      # mm/año
    "atm_pressure": (950.0, 1050.0),         # hPa
    "wind_speed": (0.0, 100.0),          # km/h
    "co2_level": (200.0, 1000.0),        # ppm
    "biomass_density": (0.0, 100.0),     # %
    "fauna_density": (0.0, 100.0)        # %
}