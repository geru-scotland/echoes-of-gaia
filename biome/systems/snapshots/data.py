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
Container class for comprehensive biome state capture.

Stores simulation time, entity states and environmental conditions;
provides structured data organization for snapshot persistence.
Includes serialization capabilities for JSON format - consolidates
terrain, entity, climate and metrics data into a unified structure.
"""

import time
from typing import Optional, Dict, Any

from shared.types import SnapshotID, TerrainData, EntityData, ClimateData, MetricsData, BiomeScoreData
from simulation.core.systems.time.time import SimulationTimeInfo


class SnapshotData:
    def __init__(self, snapshot_id: SnapshotID, simulation_time: SimulationTimeInfo):
        self.snapshot_id = snapshot_id
        self.simulation_time = simulation_time
        self.creation_timestamp = int(time.time())

        self.biome_type = None
        self.current_season = None
        self.climate_averages = None
        self.climate_analysis = None

        self.terrain_data: Optional[TerrainData] = None
        self.entities_data: Dict[int, EntityData] = {}
        self.climate_data: Optional[ClimateData] = None
        self.metrics_data: Optional[MetricsData] = None
        self.biome_score: Optional[BiomeScoreData] = None

    def set_terrain_data(self, data: TerrainData) -> None:
        self.terrain_data = data

    def set_climate_data(self, data: ClimateData) -> None:
        self.climate_averages = data

    def set_metrics_data(self, data: MetricsData) -> None:
        self.metrics_data = data

    def set_biome_score(self, data: BiomeScoreData) -> None:
        self.biome_score = data

    def set_biome_info(self, biome_type, current_season) -> None:
        self.biome_type = biome_type
        self.current_season = current_season

    def add_entity_data(self, entity_id: int, data: EntityData) -> None:
        self.entities_data[entity_id] = data

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "snapshot_id": self.snapshot_id,
            "simulation_time": self.simulation_time.to_dict(),
            "creation_timestamp": self.creation_timestamp,
        }

        if self.biome_type:
            result["biome_type"] = str(self.biome_type)

        if self.current_season:
            result["current_season"] = str(self.current_season)

        if self.climate_averages:
            result["climate_averages"] = self.climate_averages

        if self.climate_analysis:
            result["climate_analysis"] = self.climate_analysis

        if self.terrain_data:
            result["terrain"] = self.terrain_data

        if self.entities_data:
            result["entities"] = self.entities_data

        if self.climate_data:
            result["climate"] = self.climate_data

        if self.metrics_data:
            result["metrics"] = self.metrics_data

        if self.biome_score:
            result["biome_score"] = self.biome_score

        return result