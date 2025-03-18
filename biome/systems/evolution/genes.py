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

class FloraGenes:
    def __init__(self):
        self.growth_modifier = 0.0
        self.growth_efficiency = 0.0
        self.max_size = 0.0

        self.max_vitality = 0.0
        self.aging_rate = 0.0
        self.health_modifier = 0.0

        self.base_photosynthesis_efficiency = 0.0
        self.base_respiration_rate = 0.0
        self.lifespan = 0.0
        self.metabolic_activity = 0.0
        self.max_energy_reserves = 0.0

        self.cold_resistance = 0.0
        self.heat_resistance = 0.0
        self.optimal_temperature = 0.0
        # Dejo las siguientes para más adelante:
        # self.drought_resistance = 0.0
        # self.toxicity = 0.0


    def __str__(self):
        fields = {
            "growth_modifier": self.growth_modifier,
            "growth_efficiency": self.growth_efficiency,
            "max_size": self.max_size,
            "max_vitality": self.max_vitality,
            "aging_rate": self.aging_rate,
            "health_modifier": self.health_modifier,
            "base_photosynthesis_efficiency": self.base_photosynthesis_efficiency,
            "base_respiration_rate": self.base_respiration_rate,
            "lifespan": self.lifespan,
            "metabolic_activity": self.metabolic_activity,
            "max_energy_reserves": self.max_energy_reserves,
            "cold_resistance": self.cold_resistance,
            "heat_resistance": self.heat_resistance,
        }
        return ", ".join(f"{key}={value}" for key, value in fields.items())