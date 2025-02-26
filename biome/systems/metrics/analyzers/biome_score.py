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
from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Protocol, Dict, Any, List, Tuple, Set, Optional

from shared.strings import Loggers
from utils.loggers import LoggerManager


class ScoreContributor(Protocol):

    @property
    def name(self) -> str:
        ...

    @property
    def weight(self) -> float:
        ...

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        ...

class BiomeQuality(Enum):
    CRITICAL = "critical"     # 0.0 - 0.2
    UNSTABLE = "unstable"     # 0.2 - 0.4
    MODERATE = "moderate"     # 0.4 - 0.6
    HEALTHY = "healthy"       # 0.6 - 0.8
    EDEN = "eden"             # 0.8 - 1.0

    @classmethod
    def from_score(cls, score: float) -> 'BiomeQuality':
        if score < 0.2:
            return cls.CRITICAL
        elif score < 0.4:
            return cls.UNSTABLE
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.HEALTHY
        else:
            return cls.EDEN


@dataclass
class BiomeScoreResult:
    score: float
    normalized_score: float
    quality: BiomeQuality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "normalized_score": self.normalized_score,
            "quality": self.quality.value,
        }


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

        # Ratio, mido distancia al equilibrio ideal (aprox 60% flora, 40% fauna)
        ratio_flora = num_flora / total
        # TODO: Pasar a config esto
        ideal_ratio = 0.6

        # Penalizo desviaciones del ratio ideal, pero aplico mayor tolerancia a exceso de flora que de fauna
        if ratio_flora > ideal_ratio:
            # Más flora que la ideal - menos penalización
            # Sabiendo que es más, de lo que me puedo desviar (1-ideal), qué porcentaje me he desviado
            balance = 1.0 - min(1.0, (ratio_flora - ideal_ratio) / (1.0 - ideal_ratio) * 0.8)
        else:
            # Más fauna que la ideal - más penalización
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

    def calculate(self, biome_data: Dict[str, Any]) -> float:
        # Marcador para implementación futura
        # Por ahora, condiciones climáticas óptimas
        return 1.0


class BiomeScoreAnalyzer:
    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initializing BiomeScoreAnalyzer...")

        self._contributors: List[ScoreContributor] = []
        self._initialize_contributors()

    def _initialize_contributors(self) -> None:
        # TODO: Establecer weights desde configs, por contributor
        self._contributors = [
            PopulationBalanceContributor(),
            ToxicityContributor(),
            ClimateContributor()
        ]

    def calculate_score(self, biome_data: Dict[str, Any],
                        climate_data: Optional[Dict[str, Any]] = None) -> Tuple[BiomeScoreResult, Dict[str, Any]]:

        data = biome_data.copy()
        if climate_data:
            data.update(climate_data)

        total_weight: float = 0.0
        weighted_score: float = 0.0
        contributor_scores: Dict[str, int|float] = {}

        for contributor in self._contributors:
            score = contributor.calculate(data)
            contributor_scores[contributor.name] = score
            weighted_score += score * contributor.weight
            total_weight += contributor.weight

        normalized_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Escala 0-10
        score = normalized_score * 10.0

        quality = BiomeQuality.from_score(normalized_score)

        return BiomeScoreResult(
            score=round(score, 2),
            normalized_score=normalized_score,
            quality=quality
        ), contributor_scores