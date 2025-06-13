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
Biome data manager for snapshot coordination and telemetry collection.

Orchestrates snapshot system initialization with configuration support;
handles scheduled snapshot processes and telemetry data gathering.
Manages data provider integration and biome statistics collection;
supports dataset generation modes and callback-based operations.
"""

import os
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional

from simpy import Environment as simpyEnv

from biome.systems.data.providers import BiomeDataProvider
from biome.systems.snapshots.config import SnapshotConfig
from biome.systems.snapshots.system import BiomeSnapshotSystem
from config.settings import Config
from shared.enums.enums import CapturePeriod
from shared.enums.strings import Loggers
from shared.types import CallbackType
from simulation.core.systems.telemetry.datapoint import Datapoint
from utils.loggers import LoggerManager
from utils.paths import SIMULATION_DIR


class BiomeDataManager:
    def __init__(self, env: simpyEnv, config: Config, dataset_generation: bool = False):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._env = env
        self._config = config
        self._initialized = False
        self._snapshot_system = None
        self._data_provider = None
        self._snapshot_process = None
        self._dataset_generation = dataset_generation

        self._logger.debug("BiomeDataManager initialized with config")

    def configure(self, data_provider: BiomeDataProvider) -> None:
        if self._initialized:
            self._logger.warning("BiomeDataManager already initialized")
            return

        self._data_provider = data_provider
        self._initialize_snapshot_system()
        self._initialized = True
        self._logger.debug("BiomeDataManager configured with data provider")

    def _initialize_snapshot_system(self) -> None:
        try:
            snapshot_config = self._config.get("data", {}).get("snapshots", {})
            snapshot_enabled = snapshot_config.get("enabled", False)

            if not snapshot_enabled:
                self._logger.debug("Snapshot system is disabled in configuration")
                return

            config = self._create_snapshot_config(snapshot_config)

            self._snapshot_system = BiomeSnapshotSystem(
                entity_manager=self._data_provider.get_entity_provider(),
                world_map=self._data_provider.get_world_map(),
                entity_collector=self._data_provider.get_entity_collector(),
                climate_collector=self._data_provider.get_climate_collector(),
                score_analyzer=self._data_provider.get_score_analyzer(),
                config=config,
                biome_type=self._data_provider.get_biome_type(),
                climate_data_manager=self._data_provider.get_climate_data_manager(),
                dataset_generation=self._dataset_generation
            )

            capture_period = self._get_capture_period(config)
            if capture_period > 0:
                self._logger.debug(f"Starting scheduled snapshots every {capture_period} ticks")
                self._snapshot_process = self._env.process(
                    self._scheduled_snapshot_process(capture_period)
                )

            self._logger.info("Snapshot system initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize snapshot system: {e}", exc_info=True)

    def _create_snapshot_config(self, snapshot_config: Dict[str, Any]) -> SnapshotConfig:
        period_str = snapshot_config.get("period_type", "MONTHLY")
        try:
            period = CapturePeriod[period_str.upper()]
        except KeyError:
            self._logger.warning(f"Invalid capture period '{period_str}', defaulting to MONTHLY")
            period = CapturePeriod.MONTHLY

        custom_period = None
        if period == CapturePeriod.CUSTOM:
            custom_period = snapshot_config.get("custom_period", 10)

        return SnapshotConfig(
            capture_period=period,
            custom_period=custom_period,
            filename_prefix=snapshot_config.get("filename_prefix", "biome_snapshot"),
            storage_directory=Path(
                os.path.join(SIMULATION_DIR, snapshot_config.get("storage_directory", "simulation_records"))),
            pretty_print=snapshot_config.get("pretty_print", True),
            include_real_timestamp=snapshot_config.get("include_real_timestamp", True)
        )

    def _get_capture_period(self, config: SnapshotConfig) -> int:
        if config.capture_period == CapturePeriod.CUSTOM:
            if config.custom_period is None:
                self._logger.warning("Custom period not specified, defaulting to 30 ticks")
                return 30
            return config.custom_period
        return config.capture_period.value

    def _scheduled_snapshot_process(self, period: int):
        yield self._env.timeout(period)

        while True:
            self._logger.debug(f"Taking scheduled snapshot at tick {self._env.now}")
            self.capture_snapshot(self._env.now)
            yield self._env.timeout(period)

    def collect_data_for_telemetry(self, datapoint_id: int, timestamp: int) -> Optional[Datapoint]:
        if not self._initialized:
            self._logger.error("BiomeDataManager is not initialized")
            return None

        try:
            entity_collector = self._data_provider.get_entity_collector()
            score_analyzer = self._data_provider.get_score_analyzer()

            biome_statistics = entity_collector.collect_data()

            biome_score_result, contributor_scores = score_analyzer.calculate_score(biome_statistics)

            datapoint = Datapoint(
                measurement="biome_states_20",
                tags={"state_id": str(datapoint_id)},
                timestamp=timestamp,
                fields={**biome_statistics, **biome_score_result.to_dict()}
            )

            self._logger.debug(f"Collected telemetry data for datapoint {datapoint_id}")

            return datapoint
        except Exception as e:
            self._logger.exception(f"Error collecting data for telemetry: {e}")
            return None

    def capture_snapshot(self, simulation_time: int, callback: CallbackType = None) -> None:
        if not self._initialized or not self._snapshot_system:
            self._logger.warning("Cannot capture snapshot: system not initialized or disabled")
            return

        try:
            self._snapshot_system.capture_snapshot(simulation_time, callback)
            self._logger.debug(f"Snapshot capture initiated for time {simulation_time}")
        except Exception as e:
            self._logger.error(f"Failed to capture snapshot: {e}", exc_info=True)
            if callback:
                callback(None)

    def shutdown(self) -> None:
        if self._snapshot_system:
            self._logger.debug("Shutting down BiomeDataManager...")
            self._snapshot_system.shutdown()
            self._logger.debug("BiomeDataManager shutdown complete")
