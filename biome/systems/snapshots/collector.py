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
import time
from logging import Logger
from typing import Dict

import numpy as np

from biome.entities.entity import Entity
from biome.services.climate_service import ClimateService
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer
from biome.systems.metrics.analyzers.contributors import ClimateContributor
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.snapshots.data import SnapshotData
from shared.enums.enums import TerrainType, BiomeType, Season
from shared.enums.strings import Loggers
from shared.types import TerrainData, EntityData, ComponentData, ClimateData, TerrainMap
from simulation.core.systems.time.time import SimulationTimeInfo
from utils.loggers import LoggerManager


class SnapshotCollector:
    def __init__(self, biome_type: BiomeType, entity_manager: EntityProvider, climate_data_manager: ClimateDataManager, world_map: WorldMap,
                 entity_collector: EntityDataCollector, score_analyzer: BiomeScoreAnalyzer,
                 climate_collector: ClimateDataCollector = None):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._biome_type: BiomeType = biome_type
        self._entity_manager: EntityProvider = entity_manager
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self._entity_collector: EntityDataCollector = entity_collector
        self._world_map: WorldMap = world_map
        self._climate_collector: ClimateDataCollector = climate_collector
        self._score_analyzer: BiomeScoreAnalyzer = score_analyzer

    def collect_snapshot_data(self, simulation_time: int) -> SnapshotData:
        snapshot_id = f"{int(time.time())}_{simulation_time}"
        time_info = SimulationTimeInfo.from_ticks(simulation_time)

        snapshot = SnapshotData(snapshot_id, time_info)
        current_season: Season = ClimateService.get_current_season()
        snapshot.set_biome_info(self._biome_type, current_season)

        snapshot.set_terrain_data(self._collect_terrain_data())

        self._collect_entities_data(snapshot)

        self._collect_metrics_data(snapshot)

        climate_data: ClimateData = self._collect_climate_data()
        snapshot.set_climate_data(climate_data)

        self._collect_climate_analysis_data(snapshot, climate_data)

        return snapshot

    def _collect_climate_analysis_data(self, snapshot: SnapshotData, climate_data: ClimateData) -> None:
        current_season: Season = ClimateService.get_current_season()
        biome_data = {
            "biome_type": self._biome_type,
            "current_season": current_season,
            "climate_averages": climate_data
        }

        if snapshot.metrics_data:
            biome_data.update(snapshot.metrics_data)

        climate_contributor = ClimateContributor()
        climate_contributor.calculate(biome_data)

        if "climate_analysis" in biome_data:
            snapshot.climate_analysis = biome_data["climate_analysis"]

    def _collect_climate_data(self) -> ClimateData:
        try:
            climate_averages = self._climate_data_manager.get_current_month_averages()
            return climate_averages
        except Exception as e:
            self._logger.error(f"Error collecting climate data: {e}")
            return {"error": str(e)}

    def _collect_terrain_data(self) -> TerrainData:
        try:
            terrain_map: TerrainMap = self._world_map.terrain_map

            # Serializable, convierto los terrenos a int.
            terrain_int_map = terrain_map.astype(np.int8).tolist()

            terrain_types = {int(terrain): terrain.name for terrain in TerrainType}

            map_shape = terrain_map.shape
            terrain_data = {
                "terrain_map": terrain_int_map,
                "terrain_types": terrain_types,
                "map_dimensions": (map_shape[0] - 1, map_shape[1] - 1)
            }

            return terrain_data
        except Exception as e:
            self._logger.error(f"Error collecting terrain data: {e}")
            return {"error": str(e)}


    def _collect_entities_data(self, snapshot: SnapshotData) -> None:
        try:
            flora, fauna = self._entity_manager.get_entities()
            all_entities = flora + fauna

            for entity in all_entities:
                entity_data = self._collect_entity_data(entity)
                snapshot.add_entity_data(entity.get_id(), entity_data)

        except Exception as e:
            self._logger.error(f"Error collecting entities data: {e}")

    def _collect_entity_data(self, entity: Entity) -> EntityData:
        try:
            entity_data = {
                "id": entity.get_id(),
                "type": entity.get_type(),
                "species": str(entity.get_species()),
                "habitats": [str(h) for h in entity.get_habitats()],
                "state_fields": entity.get_state_fields(),
                "is_dead": entity.get_state_fields().get("general", {}).get("is_dead"),
                "components": self._collect_entity_components(entity),
                "evolution_cycle": entity.get_state_fields().get("general").get("evolution_cycle", 0),
            }

            return entity_data
        except Exception as e:
            self._logger.error(f"Error collecting data for entity {entity.get_id()}: {e}")
            return {"id": entity.get_id(), "error": str(e)}

    def _collect_entity_components(self, entity: Entity) -> Dict[str, ComponentData]:
        components_data = {}
        for component_type, component in entity.components.items():
            try:
                component_attrs = {
                    k: v for k, v in vars(component).items()
                    if not k.startswith('_') and not callable(v)
                }

                components_data[str(component_type)] = component_attrs
            except Exception as e:
                self._logger.error(f"Error collecting component {component_type} data: {e}")

        return components_data

    def _collect_metrics_data(self, snapshot: SnapshotData) -> None:
        try:
            biome_statistics = self._entity_collector.collect_data()

            biome_score_result, contributor_scores = self._score_analyzer.calculate_score(biome_statistics)

            snapshot.set_metrics_data(biome_statistics)
            snapshot.set_biome_score({
                **biome_score_result.to_dict(),
                "contributor_scores": contributor_scores
            })

        except Exception as e:
            self._logger.error(f"Error collecting metrics data: {e}")
