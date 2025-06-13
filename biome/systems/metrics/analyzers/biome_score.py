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
Biome health assessment system with weighted contributor scoring.

Evaluates overall biome quality using multiple ecological factors;
calculates normalized scores and quality classifications.
Integrates specialized contributor modules to assess different aspects -
provides comprehensive ecosystem health evaluation metrics.
"""

from dataclasses import dataclass
from enum import Enum
from logging import Logger
from typing import Any, Dict, List, Optional, Protocol, Tuple

from biome.systems.metrics.analyzers.contributors import (
    BiodiversityContributor,
    ClimateContributor,
    EcosystemHealthContributor,
    PopulationBalanceContributor,
)
from shared.enums.strings import Loggers
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
    CRITICAL = "critical"  # 0.0 - 0.2
    UNSTABLE = "unstable"  # 0.2 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.6
    HEALTHY = "healthy"  # 0.6 - 0.8
    EDEN = "eden"  # 0.8 - 1.0

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


class BiomeScoreAnalyzer:
    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initializing BiomeScoreAnalyzer...")

        self._contributors: List[ScoreContributor] = []
        self._initialize_contributors()

    def _initialize_contributors(self) -> None:
        # TODO: Establecer weights desde configs, por contributor
        self._contributors = [
            PopulationBalanceContributor(weight=1.1),
            # ToxicityContributor(weight=0.8),
            ClimateContributor(weight=1.0),
            BiodiversityContributor(weight=1.2),
            EcosystemHealthContributor(weight=1.0)
        ]

    def calculate_score(self, biome_data: Dict[str, Any],
                        climate_data: Optional[Dict[str, Any]] = None) -> Tuple[BiomeScoreResult, Dict[str, Any]]:

        data = biome_data.copy()
        if climate_data:
            data.update(climate_data)

        total_weight: float = 0.0
        weighted_score: float = 0.0
        contributor_scores: Dict[str, int | float] = {}

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
