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
from logging import Logger
from typing import Dict, List, Tuple, Set, Any

from shared.enums.strings import Loggers
from utils.loggers import LoggerManager


class BaseScoreContributor(ABC):
    def __init__(self, name: str, weight: float = 1.0):
        self._name = name
        self._weight = weight
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    @abstractmethod
    def calculate(self, biome_data: Dict[str, Any]) -> float:
        raise NotImplementedError

    def _check_data_availability(self, biome_data: Dict[str, Any], required_keys: List[str]) -> Tuple[bool, Set[str]]:
        missing = {key for key in required_keys if key not in biome_data}
        return len(missing) == 0, missing


class PopulationBalanceContributor(BaseScoreContributor):
    def __init__(self, weight: float = 1.2):
        super().__init__("population_balance", weight)

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        has_data, missing = self._check_data_availability(biome_data, ["num_flora", "num_fauna"])
        if not has_data:
            self._logger.warning(f"Missing required data for {self.name} calculation: {missing}")
            return 0.0

        num_flora = biome_data["num_flora"]
        num_fauna = biome_data["num_fauna"]
        total = num_flora + num_fauna

        if total == 0:
            return 0.0

        ratio_flora = num_flora / total
        # TODO: Pasar a config esto
        ideal_ratio = 0.6

        if ratio_flora > ideal_ratio:
            balance = 1.0 - min(1.0, (ratio_flora - ideal_ratio) / (1.0 - ideal_ratio) * 0.8)
        else:
            balance = max(0.0, ratio_flora / ideal_ratio)

        return balance


class ToxicityContributor(BaseScoreContributor):
    def __init__(self, weight: float = 0.8):
        super().__init__("toxicity", weight)
        # TODO: Pasar configs, estoy muy cansado ahora
        self._critical_threshold = 50.0

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        if "avg_toxicity" not in biome_data:
            return 1.0

        toxicity = biome_data["avg_toxicity"]

        # Convierto a una puntuación donde 0 de toxicidad = 1.0 (perfecto)
        # y toxicidad alta (>=_critical_threshold) = 0.0 (pésimo)
        normalized = max(0.0, 1.0 - (toxicity / self._critical_threshold))

        return normalized


class ClimateContributor(BaseScoreContributor):
    def __init__(self, weight: float = 1.3):
        super().__init__("climate", weight)
        self._climate_factors = {
            "temperature": 0.4,
            "humidity": 0.3,
            "precipitation": 0.3
        }

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        biome_type = biome_data.get("biome_type")

        optimal_ranges = {
            "temperature": (15.0, 25.0),
            "humidity": (40.0, 70.0),
            "precipitation": (30.0, 80.0)
        }

        if biome_type:
            try:
                from shared.stores.biome_store import BiomeStore
                biome_info = BiomeStore.biomes.get(biome_type, {})
                environmental_factors = biome_info.get("environmental_factors", {})

                if "temperature" in environmental_factors:
                    optimal_ranges["temperature"] = (
                        environmental_factors["temperature"].get("min", 15.0),
                        environmental_factors["temperature"].get("max", 25.0)
                    )
                if "humidity" in environmental_factors:
                    optimal_ranges["humidity"] = (
                        environmental_factors["humidity"].get("min", 40.0),
                        environmental_factors["humidity"].get("max", 70.0)
                    )
                if "precipitation" in environmental_factors:
                    optimal_ranges["precipitation"] = (
                        environmental_factors["precipitation"].get("min", 30.0),
                        environmental_factors["precipitation"].get("max", 80.0)
                    )
            except ImportError:
                self._logger.warning("BiomeStore not available for climate range lookup")
            except Exception as e:
                self._logger.warning(f"Error retrieving climate ranges from BiomeStore: {e}")

        climate_averages = {}
        if "climate_averages" in biome_data:
            climate_averages = biome_data["climate_averages"]
        elif "climate" in biome_data and "current_season" in biome_data:
            climate = biome_data["climate"]
            climate_averages = {
                "avg_temperature": climate.get("temperature", 20.0),
                "avg_humidity": climate.get("humidity", 50.0),
                "avg_precipitation": climate.get("precipitation", 40.0)
            }

        if not climate_averages:
            return 0.7

        climate_score = 0.0
        factor_count = 0

        climate_analysis = {
            "factors": {},
        }

        for factor, weight in self._climate_factors.items():
            factor_key = f"avg_{factor}"
            if factor_key in climate_averages:
                value = climate_averages[factor_key]
                min_val, max_val = optimal_ranges[factor]

                if min_val <= value <= max_val:
                    factor_score = 1.0
                    status = "optimal"
                elif value < min_val:
                    distance = min_val - value
                    max_distance = min_val
                    factor_score = max(0.0, 1.0 - (distance / max_distance))
                    status = "too_low"
                else:
                    distance = value - max_val
                    max_distance = 100.0 - max_val
                    factor_score = max(0.0, 1.0 - (distance / max_distance))
                    status = "too_high"

                climate_score += factor_score * weight
                factor_count += weight

                climate_analysis["factors"][factor] = {
                    "value": value,
                    "optimal_range": (min_val, max_val),
                    "status": status
                }

        if factor_count == 0:
            return 0.5

        final_score = climate_score / factor_count
        climate_analysis["overall_score"] = final_score

        biome_data["climate_analysis"] = climate_analysis

        return final_score


class BiodiversityContributor(BaseScoreContributor):
    def __init__(self, weight: float = 0.9):
        super().__init__("biodiversity", weight)

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        has_data, missing = self._check_data_availability(biome_data, ["num_flora", "num_fauna"])
        if not has_data:
            self._logger.warning(f"Missing required data for {self.name} calculation: {missing}")
            return 0.5

        num_flora = biome_data["num_flora"]
        num_fauna = biome_data["num_fauna"]
        total = num_flora + num_fauna

        if total == 0:
            return 0.0

        shannon_biodiversity = biome_data.get("biodiversity_index")

        if shannon_biodiversity is None:
            flora_species_count = biome_data.get("flora_species_count", 1)
            fauna_species_count = biome_data.get("fauna_species_count", 1)

            if "flora_species_count" not in biome_data:
                flora_species_count = min(5, max(1, num_flora // 3))
            if "fauna_species_count" not in biome_data:
                fauna_species_count = min(4, max(1, num_fauna // 2))

            total_species = flora_species_count + fauna_species_count

            biodiversity_score = min(1.0, total_species / 8.0)

            if total_species < 3:
                biodiversity_score *= 0.7
        else:
            biodiversity_score = shannon_biodiversity
            self._logger.debug(f"Using Shannon-Wiener biodiversity index: {biodiversity_score:.3f}")

        prey_population = biome_data.get("prey_population", 0)
        predator_population = biome_data.get("predator_population", 0)

        if num_fauna > 0 and prey_population == 0 and predator_population == 0:
            self._logger.debug("No prey/predator data available, using default calculations")
            prey_population = num_fauna * 0.7
            predator_population = num_fauna * 0.3

        if prey_population > 0:
            pred_prey_ratio = predator_population / prey_population
        else:
            # Un poco bruto, pero pongo inf para edge case, si no hay presas y si depredadores, muy desequilibrado
            pred_prey_ratio = float('inf') if predator_population > 0 else 0.0

        optimal_min_ratio = 0.1
        optimal_max_ratio = 0.3

        if pred_prey_ratio == float('inf'):
            predator_prey_balance_score = 0.1
            self._logger.debug(f"Critical imbalance: only predators present")
        elif pred_prey_ratio == 0.0 and predator_population == 0 and prey_population > 0:
            predator_prey_balance_score = 0.4
            self._logger.debug(f"Imbalance: only prey present")
        elif pred_prey_ratio < optimal_min_ratio:
            balance_factor = pred_prey_ratio / optimal_min_ratio
            predator_prey_balance_score = 0.4 + (balance_factor * 0.6)
            self._logger.debug(
                f"Too few predators: ratio {pred_prey_ratio:.3f}, score {predator_prey_balance_score:.2f}")
        elif pred_prey_ratio > optimal_max_ratio:
            excess_factor = min(1.0, (pred_prey_ratio - optimal_max_ratio) / 2.0)
            predator_prey_balance_score = 1.0 - (excess_factor * 0.8)
            self._logger.debug(
                f"Too many predators: ratio {pred_prey_ratio:.3f}, score {predator_prey_balance_score:.2f}")
        else:
            optimal_mid_ratio = (optimal_min_ratio + optimal_max_ratio) / 2
            distance_from_mid = abs(pred_prey_ratio - optimal_mid_ratio) / (optimal_max_ratio - optimal_min_ratio)
            predator_prey_balance_score = 1.0 - (distance_from_mid * 0.2)
            self._logger.debug(
                f"Optimal predator-prey ratio: {pred_prey_ratio:.3f}, score {predator_prey_balance_score:.2f}")

        # Nota: Cacharrear y quizá dar algo más de peso al predator_prey_balance_score, que me es más útil por ahora.
        biodiversity_weight = 0.5
        balance_weight = 0.5
        final_score = (biodiversity_score * biodiversity_weight) + (predator_prey_balance_score * balance_weight)

        self._logger.debug(
            f"Biodiversity score: {biodiversity_score:.2f}, Balance score: {predator_prey_balance_score:.2f}, Final: {final_score:.2f}")

        return final_score


class EcosystemHealthContributor(BaseScoreContributor):
    def __init__(self, weight: float = 1.1):
        super().__init__("ecosystem_health", weight)

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        health_score = 0.0
        factors_count = 0

        if "avg_stress_level" in biome_data:
            stress_level = min(100.0, biome_data["avg_stress_level"])
            stress_factor = 1.0 - (stress_level / 100.0)
            health_score += stress_factor * 1.5
            factors_count += 1.5
            self._logger.debug(f"Stress factor: {stress_factor:.2f} from level: {stress_level:.2f}")

        if "avg_size" in biome_data:
            size = biome_data["avg_size"]
            size_factor = min(1.0, size / 2.0) if size <= 2.0 else max(0.0, 1.0 - (size - 2.0) / 3.0)
            health_score += size_factor
            factors_count += 1
            self._logger.debug(f"Size factor: {size_factor:.2f} from size: {size:.2f}")

        if "climate_adaptation" in biome_data:
            adaptation = biome_data["climate_adaptation"]
            health_score += adaptation * 1.0
            factors_count += 1.0
            self._logger.debug(f"Climate adaptation factor: {adaptation:.2f}")

        if "entity_balance" in biome_data:
            balance = biome_data["entity_balance"]
            health_score += balance * 1.0
            factors_count += 1.0
            self._logger.debug(f"Species balance factor: {balance:.2f}")

        if "avg_toxicity" in biome_data:
            toxicity = min(100.0, biome_data["avg_toxicity"])
            toxicity_factor = 1.0 - (toxicity / 100.0)
            health_score += toxicity_factor * 0.8
            factors_count += 0.8
            self._logger.debug(f"Toxicity factor: {toxicity_factor:.2f} from toxicity: {toxicity:.2f}")

        if factors_count > 0:
            health_score /= factors_count

        health_score = min(1.0, health_score)
        return health_score
