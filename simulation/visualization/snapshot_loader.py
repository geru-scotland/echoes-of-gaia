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
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from simulation.visualization.types import SnapshotData


class SnapshotLoader:
    def __init__(self, snapshot_file: Path):
        self._logger = logging.getLogger("snapshot_loader")
        self._file_path = snapshot_file
        self._snapshots: List[SnapshotData] = []
        self._current_index: int = 0
        self._common_terrain = None

    def load(self) -> bool:
        try:
            self._logger.info(f"Loading snapshots from {self._file_path}")

            filename_parts = self._file_path.stem.split('_')
            if len(filename_parts) >= 3:
                timestamp_part = '_'.join(filename_parts[2:])

                terrain_path = self._file_path.parent / f"{filename_parts[0]}_terrain_{timestamp_part}.json"
                self._logger.info(f"Looking for terrain file: {terrain_path}")

                if terrain_path.exists():
                    with open(terrain_path, 'r') as f:
                        self._common_terrain = json.load(f)
                        self._logger.info(f"Loaded terrain data from {terrain_path}")
                else:
                    terrain_files = list(self._file_path.parent.glob(f"{filename_parts[0]}_terrain_*.json"))
                    if terrain_files:
                        latest_terrain = max(terrain_files, key=lambda p: p.stat().st_mtime)
                        with open(latest_terrain, 'r') as f:
                            self._common_terrain = json.load(f)
                            self._logger.info(f"Loaded terrain data from {latest_terrain}")
                    else:
                        self._logger.warning("No terrain file found")

            with open(self._file_path, 'r') as f:
                self._snapshots = json.load(f)

            self._current_index = 0
            self._logger.info(f"Successfully loaded {len(self._snapshots)} snapshots")
            return True
        except Exception as e:
            self._logger.error(f"Error loading snapshots: {e}")
            return False

    def get_current_snapshot(self) -> Optional[SnapshotData]:
        if not self._snapshots:
            return None

        snapshot = self._snapshots[self._current_index]

        if self._common_terrain is not None and "terrain" not in snapshot:
            snapshot = snapshot.copy()
            snapshot["terrain"] = self._common_terrain

        return snapshot

    def next_snapshot(self) -> Optional[SnapshotData]:
        if not self._snapshots:
            return None

        if self._current_index < len(self._snapshots) - 1:
            self._current_index += 1
            return self.get_current_snapshot()

        return None

    def previous_snapshot(self) -> Optional[SnapshotData]:
        if not self._snapshots:
            return None

        if self._current_index > 0:
            self._current_index -= 1
            return self.get_current_snapshot()

        return None

    def go_to_snapshot(self, index: int) -> Optional[SnapshotData]:
        if not self._snapshots:
            return None

        if 0 <= index < len(self._snapshots):
            self._current_index = index
            return self.get_current_snapshot()

        return None

    def get_snapshot_count(self) -> int:
        return len(self._snapshots)

    def get_current_index(self) -> int:
        return self._current_index