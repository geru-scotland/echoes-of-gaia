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

import json
import time
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import Any, Dict

import numpy as np

from biome.systems.snapshots.config import SnapshotConfig
from biome.systems.snapshots.data import SnapshotData
from shared.enums import CaptureFormat
from shared.strings import Loggers
from shared.types import CallbackType
from utils.loggers import LoggerManager


class SnapshotStorage:
    def __init__(self, config: SnapshotConfig):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._config = config
        self._ensure_storage_directory()
        self._terrain_saved = False
        self._snapshot_filepath = None
        self._snapshot_file_exists = False
        self._snapshot_counter = 1

        self._logger.info(f"Snapshot storage initialized with directory: {self._config.storage_directory}")

    def _ensure_storage_directory(self) -> None:
        try:
            self._config.storage_directory.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Ensured storage directory exists: {self._config.storage_directory}")
        except Exception as e:
            self._logger.error(f"Failed to create storage directory: {e}")
            raise

    def save_snapshot(self, snapshot: SnapshotData, callback: CallbackType = None) -> None:
        self._logger.debug(f"Saving snapshot synchronously")
        filepath = None

        try:
            snapshot.snapshot_id = str(self._snapshot_counter)
            self._snapshot_counter += 1

            filepath = self._save_snapshot_internal(snapshot)
            self._logger.debug(f"Successfully saved snapshot to {filepath}")
        except Exception as e:
            self._logger.error(f"Error saving snapshot: {e}")
        finally:
            if callback:
                callback(filepath)

    def _save_snapshot_internal(self, snapshot: SnapshotData) -> Path:
        try:
            if not self._terrain_saved and snapshot.terrain_data:
                self._save_terrain_data(snapshot.terrain_data)
                self._terrain_saved = True

            if not self._snapshot_filepath:
                filename = self._generate_filename()
                self._snapshot_filepath = self._config.storage_directory / filename
                self._snapshot_file_exists = False
                self._logger.debug(f"Created new snapshot file: {self._snapshot_filepath}")

            snapshot_data = snapshot.to_dict()
            if 'terrain' in snapshot_data:
                del snapshot_data['terrain']

            mode = 'w' if not self._snapshot_file_exists else 'a'

            with open(self._snapshot_filepath, mode) as f:
                if self._config.capture_format == CaptureFormat.JSON:
                    if not self._snapshot_file_exists:
                        f.write('[\n')
                    else:
                        f.write(',\n')

                    indent = 2 if self._config.pretty_print else None
                    json.dump(snapshot_data, f, indent=indent, default=self._json_serializer)
                elif self._config.capture_format == CaptureFormat.JSONL:
                    f.write(json.dumps(snapshot_data, default=self._json_serializer) + '\n')

            self._snapshot_file_exists = True
            return self._snapshot_filepath

        except Exception as e:
            self._logger.error(f"Failed to save snapshot: {e}")
            raise

    def _save_terrain_data(self, terrain_data: Dict) -> Path:
        try:
            current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            terrain_filename = f"{self._config.terrain_prefix}_terrain_{current_time}.json"
            terrain_filepath = self._config.storage_directory / terrain_filename

            with open(terrain_filepath, 'w') as f:
                json.dump(terrain_data, f,
                          indent=2 if self._config.pretty_print else None,
                          default=self._json_serializer)

            self._logger.debug(f"Saved terrain data to {terrain_filepath}")
            self._terrain_filename = terrain_filename
            return terrain_filepath

        except Exception as e:
            self._logger.error(f"Failed to save terrain data: {e}")
            raise

    def _generate_filename(self) -> str:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        extension = self._config.capture_format.value
        return f"{self._config.filename_prefix}_{current_time}.{extension}"

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return str(obj)

    def shutdown(self) -> None:
        self._logger.info("Shutting down snapshot storage...")

        if self._snapshot_file_exists and self._snapshot_filepath:
            if self._config.capture_format == CaptureFormat.JSON:
                try:
                    with open(self._snapshot_filepath, 'a') as f:
                        f.write('\n]')
                    self._logger.info(f"Closed JSON array in {self._snapshot_filepath}")
                except Exception as e:
                    self._logger.error(f"Error closing JSON array: {e}")

        self._logger.info("Snapshot storage shutdown complete")