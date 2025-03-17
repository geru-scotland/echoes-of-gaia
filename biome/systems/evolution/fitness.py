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


def compute_fitness(flora_genes, climate_data):
    fitness: float = 0.0

    avg_temperature = climate_data['temperature_ema'].iloc[-1]
    avg_humidity = climate_data['humidity_ema'].iloc[-1]
    avg_precipitation = climate_data['precipitation_ema'].iloc[-1]

    # 1. Adaptación a temperatura
    temperature_score: float = 0.0
    if avg_temperature < 0:  # Frío
        temperature_score = flora_genes.cold_resistance * 5.0
    elif avg_temperature > 30:  # Calor
        temperature_score = flora_genes.heat_resistance * 5.0
    else:
        temperature_score = 3.0

    # 2. Adaptación a humedad
    humidity_score: float = 0.0
    if avg_humidity < 30:  # Seco
        # Bajo condiciones secas:
        # - Menor respiración es mejor - conserva agua
        humidity_score = 5.0 * (1.0 - flora_genes.base_respiration_rate)
        # - Eficiencia fotosintética alta sigue siendo importante
        humidity_score += 2.0 * flora_genes.base_photosynthesis_efficiency
    elif avg_humidity > 70:  # Húmedo
        # Alto metabolismo - aprovecha la humedad
        humidity_score = 3.0 * flora_genes.metabolic_activity
        # Alta fotosíntesis - aprovecha la humedad
        humidity_score += 3.0 * flora_genes.base_photosynthesis_efficiency
    else:
        humidity_score = 3.0

    # 3. Adaptación a precipitación
    precipitation_score: float = 0.0
    if avg_precipitation < 30:  # Muy seco
        # Premiar eficiencia fotosintética - aprovecha la poca agua
        precipitation_score = 4.0 * flora_genes.base_photosynthesis_efficiency
        # Premiar reservas de energía - aguanta sequías
        precipitation_score += 3.0 * (flora_genes.max_energy_reserves / 200.0)
    elif avg_precipitation > 100:  # Muy lluvioso
        # Premio crecimiento rápido (si crece rápido, aprovecha realmente lluvia)
        precipitation_score = 3.0 * flora_genes.growth_modifier
        # Si llega al tamño máximo, aprovechamiento total
        precipitation_score += 2.0 * (flora_genes.max_size / 5.0)
    else:
        precipitation_score = 3.0

    # 4. Características de supervivencia general
    survival_score: float = 0.0
    # Vitalidad alta; bueno
    survival_score += 2.0 * (flora_genes.max_vitality / 200.0)
    # # Longevidad;
    # survival_score += 1.0 * (flora_genes.lifespan / 20.0)
    # Envejecimiento lento es mejor
    survival_score += 1.5 * (1.0 - flora_genes.aging_rate / 2.0)

    # 5. Características de eficiencia energética
    energy_score: float = 0.0
    # Alta eficiencia de crecimiento
    energy_score += 2.0 * flora_genes.growth_efficiency
    # Buen modificador de salud
    energy_score += 1.5 * flora_genes.health_modifier
    # Actividad metabólica óptima (penalizo extremos)
    optimal_metabolic = 1.0 - abs(flora_genes.metabolic_activity - 0.8) / 0.8
    energy_score += 2.0 * optimal_metabolic

    # Penalizo por respiración excesiva (consumo energético)
    if flora_genes.base_respiration_rate > 0.1:
        energy_score -= (flora_genes.base_respiration_rate - 0.1) * 10.0

    # Combinar todas las puntuaciones
    fitness = temperature_score + humidity_score + precipitation_score + survival_score + energy_score

    # TODO: Poner logger especifico, o al menos de agente de evolución
    # print(f"Climate: Temp={avg_temperature:.1f}°C, Humidity={avg_humidity:.1f}%, Precip={avg_precipitation:.1f}mm")
    # print(f"Fitness breakdown: Temp={temperature_score:.1f}, Humidity={humidity_score:.1f}, "
    #       f"Precip={precipitation_score:.1f}, Survival={survival_score:.1f}, Energy={energy_score:.1f}")
    # print(f"Total fitness: {fitness:.2f}")

    return fitness