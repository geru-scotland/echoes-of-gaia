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
import threading
import time
from enum import Enum
from logging import Logger
from pathlib import Path
from queue import Queue, Empty
from typing import Any

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
        self._save_queue = Queue()
        self._worker_thread = threading.Thread(
            target=self._save_worker,
            daemon=True,
            name="SnapshotSaveWorkerThread"
        )
        self._worker_thread.start()
        self._running = True
        self._logger.info(f"Snapshot storage initialized with directory: {self._config.storage_directory}")

    def _ensure_storage_directory(self) -> None:
        try:
            self._config.storage_directory.mkdir(parents=True, exist_ok=True)
            self._logger.error(f"Ensured storage directory exists: {self._config.storage_directory}")
        except Exception as e:
            self._logger.error(f"Failed to create storage directory: {e}")
            raise

    def _save_worker(self):
        self._logger.info("Snapshot save worker thread started")
        snapshot_counter = 1

        while self._running:
            try:
                snapshot, callback = self._save_queue.get(timeout=1)
                filepath = None

                try:
                    snapshot.snapshot_id = str(snapshot_counter)
                    snapshot_counter += 1

                    filepath = self._save_snapshot_internal(snapshot)
                    self._logger.info(f"Successfully saved snapshot to {filepath}")
                except Exception as e:
                    self._logger.error(f"Error saving snapshot: {e}")
                finally:
                    if callback:
                        callback(filepath)
                    self._save_queue.task_done()
            except Empty:
                pass

        self._logger.info("Snapshot save worker thread stopping")

        if hasattr(self, '_snapshot_file_exists') and self._snapshot_file_exists:
            if self._config.capture_format == CaptureFormat.JSON:
                # Leer todo el archivo
                with open(self._snapshot_filepath, 'r') as f:
                    content = f.read()

                if content.endswith(']'):
                    content = content[:-1]
                elif content.endswith(',\n'):
                    content = content[:-2]
                elif content.endswith('\n'):
                    content = content[:-1]

                try:
                    json_content = content + ']'
                    snapshots = json.loads(json_content)

                    if snapshots and len(snapshots) > 0:
                        terrain = snapshots[0].get("terrain")
                        if terrain:
                            optimized_snapshots = []
                            for snapshot in snapshots:
                                s_copy = snapshot.copy()
                                if "terrain" in s_copy:
                                    del s_copy["terrain"]
                                optimized_snapshots.append(s_copy)

                            optimized_content = json.dumps({
                                "terrain": terrain,
                                "snapshots": optimized_snapshots
                            }, indent=2 if self._config.pretty_print else None, default=self._json_serializer)

                            with open(self._snapshot_filepath, 'w') as f:
                                f.write(optimized_content)

                            self._logger.info(f"Optimized snapshots in {self._snapshot_filepath}")
                            return
                except Exception as e:
                    self._logger.warning(f"Could not optimize snapshots: {e}")

                with open(self._snapshot_filepath, 'a') as f:
                    f.write('\n]')
                self._logger.info(f"Closed JSON array in {self._snapshot_filepath}")

    def save_snapshot(self, snapshot: SnapshotData, callback: CallbackType = None) -> None:
        self._logger.debug(f"Enqueueing snapshot {snapshot.snapshot_id} for saving")
        self._save_queue.put((snapshot, callback))

    def _save_snapshot_internal(self, snapshot: SnapshotData) -> Path:
        try:
            if not hasattr(self, '_snapshot_filepath'):
                filename = self._generate_filename()
                self._snapshot_filepath = self._config.storage_directory / filename
                self._snapshot_file_exists = False
                self._logger.error(f"Created new snapshot file: {self._snapshot_filepath}")

            data = snapshot.to_dict()

            mode = 'w' if not self._snapshot_file_exists else 'a'

            with open(self._snapshot_filepath, mode) as f:
                if self._config.capture_format == CaptureFormat.JSON:
                    if not self._snapshot_file_exists:
                        f.write('[\n')
                    else:
                        f.write(',\n')

                    indent = 2 if self._config.pretty_print else None
                    json.dump(data, f, indent=indent, default=self._json_serializer)

                elif self._config.capture_format == CaptureFormat.JSONL:
                    f.write(json.dumps(data, default=self._json_serializer) + '\n')


            self._snapshot_file_exists = True
            return self._snapshot_filepath

        except Exception as e:
            self._logger.error(f"Failed to save snapshot: {e}")
            raise

    def _generate_filename(self) -> str:
        current_time = time.localtime()
        formatted_time = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)

        extension = self._config.capture_format.value
        filename = f"{self._config.filename_prefix}_{formatted_time}.{extension}"
        return filename

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int8) or isinstance(obj, np.int16) or isinstance(obj, np.int32) or isinstance(obj,
                                                                                                            np.int64):
            return int(obj)
        if isinstance(obj, np.float16) or isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return str(obj)

    def shutdown(self):
        self._logger.info("Shutting down snapshot storage...")
        self._running = False

        if not self._save_queue.empty():
            self._logger.info("Waiting for pending snapshot saves to complete...")
            self._logger.info(f"There are {self._save_queue.qsize()} pending snapshot saves")
            try:
                self._save_queue.join_nowait()  # Esta línea es clave: no usar join() que bloquea
            except:
                self._logger.warning("Could not wait for all tasks to complete")

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3)
            if self._worker_thread.is_alive():
                self._logger.warning("Snapshot worker thread did not terminate gracefully")
            else:
                self._logger.info("Snapshot worker thread terminated gracefully")

        self._logger.info("Snapshot storage shutdown complete")

