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
from logging import Logger

from biome.systems.managers.entity_manager import EntityManager
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.snapshots.collector import SnapshotCollector
from biome.systems.snapshots.config import SnapshotConfig
from biome.systems.snapshots.data import SnapshotData
from biome.systems.snapshots.storage import SnapshotStorage
from shared.enums.strings import Loggers
from shared.types import CallbackType
from utils.loggers import LoggerManager


class BiomeSnapshotSystem:
    def __init__(self, entity_manager: EntityManager, world_map: WorldMap,
                 entity_collector: EntityDataCollector, score_analyzer: BiomeScoreAnalyzer,
                 config: SnapshotConfig):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._logger.info("Initializing BiomeSnapshotSystem...")

        self._entity_manager = entity_manager
        self._world_map = world_map
        self._entity_collector = entity_collector
        self._score_analyzer = score_analyzer
        self._config = config

        self._collector = SnapshotCollector(
            entity_manager, world_map, entity_collector, score_analyzer
        )
        self._storage = SnapshotStorage(self._config)

        self._logger.info("BiomeSnapshotSystem initialized successfully")

    def collect_snapshot_data(self, simulation_time: int) -> SnapshotData:
        return self._collector.collect_snapshot_data(simulation_time)

    def capture_snapshot(self, simulation_time: int, callback: CallbackType = None) -> None:
        try:
            snapshot = self.collect_snapshot_data(simulation_time)
            self._storage.save_snapshot(snapshot, callback)
            self._logger.debug(f"Snapshot capture initiated for time {simulation_time}")
        except Exception as e:
            self._logger.error(f"Failed to capture snapshot: {e}", exc_info=True)
            if callback:
                callback(None)

    def shutdown(self) -> None:
        if hasattr(self, '_storage'):
            self._storage.shutdown()