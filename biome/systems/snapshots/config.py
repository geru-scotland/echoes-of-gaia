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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional

from shared.enums import CapturePeriod, CaptureFormat, CaptureType
from utils.paths import SIMULATION_DIR


@dataclass
class SnapshotConfig:
    capture_period: CapturePeriod = CapturePeriod.MONTHLY
    capture_format: CaptureFormat = CaptureFormat.JSON
    storage_directory: Path = field(default_factory=lambda: Path(SIMULATION_DIR) / "simulation_records")
    capture_types: Set[CaptureType] = field(default_factory=lambda: {CaptureType.FULL})
    filename_prefix: str = "biome_snapshot"
    custom_period: Optional[int] = None
    include_real_timestamp: bool = True
    pretty_print: bool = True

    def __post_init__(self):
        if self.capture_period == CapturePeriod.CUSTOM and self.custom_period is None:
            raise ValueError("custom_period must be specified when capture_period is CUSTOM")
