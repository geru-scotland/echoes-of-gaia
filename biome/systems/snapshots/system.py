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
import itertools
from logging import Logger

from biome.services.climate_service import ClimateService
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer
from biome.systems.metrics.analyzers.contributors import ClimateContributor
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.snapshots.collector import SnapshotCollector
from biome.systems.snapshots.config import SnapshotConfig
from biome.systems.snapshots.data import SnapshotData
from biome.systems.snapshots.storage import SnapshotStorage
from shared.enums.strings import Loggers
from shared.types import CallbackType
from utils.loggers import LoggerManager


class BiomeSnapshotSystem:
    def __init__(self, entity_manager: EntityProvider, world_map: WorldMap,
                 entity_collector: EntityDataCollector, score_analyzer: BiomeScoreAnalyzer,
                 config: SnapshotConfig, climate_collector: ClimateDataCollector = None,
                 biome_type=None, climate_data_manager: ClimateDataManager = None, dataset_generation: bool = False):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initializing BiomeSnapshotSystem...")

        self._entity_manager: EntityProvider = entity_manager
        self._world_map: WorldMap = world_map
        self._entity_collector: EntityDataCollector = entity_collector
        self._climate_collector: ClimateDataCollector = climate_collector
        self._score_analyzer: BiomeScoreAnalyzer = score_analyzer
        self._config: SnapshotConfig = config
        self._dataset_generation: bool = dataset_generation

        self._climate_data_manager = climate_data_manager

        self._collector = SnapshotCollector(
            biome_type, entity_manager, climate_data_manager, world_map, entity_collector, score_analyzer,
            dataset_generation
        )
        self._storage = SnapshotStorage(self._config, dataset_generation)
        self._snapshot_generator: itertools.count[int] = itertools.count(0)

        self._logger.info("BiomeSnapshotSystem initialized successfully")

    def capture_snapshot(self, simulation_time: int, callback: CallbackType = None) -> None:
        try:
            snapshot_id: int = next(self._snapshot_generator)
            snapshot_data = self._collector.collect_snapshot_data(simulation_time, snapshot_id)

            # TODO: Solo si activo en configs
            neurosymbolic_data = self._collector.get_neurosymbolic_data()

            if self._dataset_generation and neurosymbolic_data:
                self._storage.save_neurosymbolic_data(neurosymbolic_data, snapshot_id)

            self._storage.save_snapshot(snapshot_data, callback)

            self._logger.debug(f"Snapshot and neurosymbolic data captured for time {simulation_time}")
        except Exception as e:
            self._logger.error(f"Failed to capture snapshot or neurosymbolic data: {e}", exc_info=True)
            if callback:
                callback(None)

    def shutdown(self) -> None:
        if hasattr(self, '_storage'):
            self._storage.shutdown()
