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
import math

from biome.systems.evolution.genes.fauna_genes import FaunaGenes
from biome.systems.evolution.genes.flora_genes import FloraGenes
from biome.systems.evolution.genes.genes import Genes
from shared.math.constants import epsilon

def compute_fitness(genes: Genes, climate_data) -> float:
    """Calcula el fitness para cualquier tipo de genes."""
    if isinstance(genes, FloraGenes):
        return compute_flora_fitness(genes, climate_data)
    elif isinstance(genes, FaunaGenes):
        return compute_fauna_fitness(genes, climate_data)
    else:
        raise ValueError(f"Unsupported genes type: {type(genes)}")


def compute_fauna_fitness(fauna_genes: FaunaGenes, climate_data) -> float:
    fitness: float = 0.0

    avg_temperature = climate_data['temperature_ema'].iloc[-1]
    avg_humidity = climate_data['humidity_ema'].iloc[-1]
    avg_precipitation = climate_data['precipitation_ema'].iloc[-1]

    # 1. Adaptación a temperatura (similar a flora)
    optimal_temperature = fauna_genes.optimal_temperature
    temperature_distance = avg_temperature - optimal_temperature

    if temperature_distance < 0:  # Frío
        sigma_cold = 15.0 / (1.0 - fauna_genes.cold_resistance + 1e-6)
        stress_factor = math.exp(-(temperature_distance ** 2) / (2 * sigma_cold ** 2))
    else:  # Calor
        sigma_heat = 15.0 / (1.0 - fauna_genes.heat_resistance + 1e-6)
        stress_factor = math.exp(-(temperature_distance ** 2) / (2 * sigma_heat ** 2))

    temperature_score = 5.0 * stress_factor

    # 2. Capacidad de supervivencia, especificos de fauna, ya iré poniendo.
    # mobility_score = 3.0 * fauna_genes.speed
    # survival_score = 2.0 * fauna_genes.predator_avoidance
    # foraging_score = 2.5 * fauna_genes.foraging_efficiency

    # 3. Salud y vitalidad (compartido con flora)
    health_score = 2.0 * (fauna_genes.max_vitality / 200.0)
    aging_score = 1.5 * (1.0 - fauna_genes.aging_rate / 2.0)

    fitness = (temperature_score + health_score + aging_score)

    return fitness

def compute_flora_fitness(flora_genes, climate_data):
    fitness: float = 0.0

    avg_temperature = climate_data['temperature_ema'].iloc[-1]
    avg_humidity = climate_data['humidity_ema'].iloc[-1]
    avg_precipitation = climate_data['precipitation_ema'].iloc[-1]

    # 1. Adaptación a temperatura
    optimal_temperature = flora_genes.optimal_temperature

    temperature_distance = avg_temperature - optimal_temperature

    # Si resistencia grande, 5.0 se divide entre txiki - sigma grande (distri. grande, muy alta tolerancia a diferencia de º)
    if temperature_distance < 0:  # Frío
        sigma_cold = 15.0 / (1.0 - flora_genes.cold_resistance + 1e-6)
        stress_factor = math.exp(-(temperature_distance ** 2) / (2 * sigma_cold ** 2))
    else:  # Calor
        sigma_heat = 15.0 / (1.0 - flora_genes.heat_resistance + 1e-6)
        stress_factor = math.exp(-(temperature_distance ** 2) / (2 * sigma_heat ** 2))

    temperature_score = 5.0 * stress_factor

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

    fitness = temperature_score + humidity_score + precipitation_score + survival_score + energy_score

    # Penalización NUEVA: evita respiration_rate demasiado bajo
    MIN_RESPIRATION_THRESHOLD = 0.18
    if flora_genes.base_respiration_rate < MIN_RESPIRATION_THRESHOLD:
        respiration_penalty = (MIN_RESPIRATION_THRESHOLD - flora_genes.base_respiration_rate) * 30.0
        fitness -= respiration_penalty
    light_availability = 0.5
    temperature_modifier = 0.7
    water_modifier = 0.6

    effective_photosynthesis = (
            flora_genes.base_photosynthesis_efficiency *
            light_availability *
            temperature_modifier *
            water_modifier *
            flora_genes.metabolic_activity
    )

    effective_respiration = flora_genes.base_respiration_rate * flora_genes.metabolic_activity

    if effective_photosynthesis <= effective_respiration:
        return fitness * 0.1

    energy_ratio = effective_photosynthesis / effective_respiration
    if energy_ratio > 1.0:
        energy_bonus = min(5.0, energy_ratio - 1.0) * 2.0
        fitness += energy_bonus

    photosynthesis = flora_genes.base_photosynthesis_efficiency
    respiration = flora_genes.base_respiration_rate

    max_allowed_ratio = 3.0
    ratio = photosynthesis / (respiration + epsilon)

    if ratio > max_allowed_ratio:
        extreme_ratio_penalty = (ratio - max_allowed_ratio) * 2.0
        fitness -= extreme_ratio_penalty

    nutrition_score = 0.0

    # La absorción de nutrientes - mucho mejor en condiciones húmedas
    if avg_humidity > 60:
        nutrition_score += 3.0 * flora_genes.nutrient_absorption_rate

    # Las micorrizas - he leído que mejores en condiciones secas
    if avg_precipitation < 50:
        nutrition_score += 2.5 * flora_genes.mycorrhizal_rate * 10  # Rate es muy txiki, lo multiplico

    # Valor nutritivo alto penalizo (ojo... revisar bien) -  la idea es
    # supervivencia (atrae herbívoros)
    # nutrition_score -= 0.5 * flora_genes.base_nutritive_value

    # La toxicidad es defensa (hormésis de nuevo, un poco - gud, mucho - bad), pero perjudicial en exceso
    if flora_genes.base_toxicity < 0.4:
        nutrition_score += 1.0 * flora_genes.base_toxicity
    else:
        nutrition_score -= (flora_genes.base_toxicity - 0.4) * 2.0

    fitness += nutrition_score


    return fitness