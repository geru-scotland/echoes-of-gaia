"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
import argparse
import os
import sys
from pathlib import Path

from simulation.visualization.snapshot_viewer import SnapshotViewer
from simulation.visualization.types import ViewerConfig
from utils.loggers import LoggerManager
from utils.paths import SIMULATION_DIR

TERRAIN_COLORS = {
    0: (15, 42, 68),    # OCEAN_DEEP
    1: (40, 84, 116),   # OCEAN_MID
    2: (72, 123, 153),  # OCEAN_SHALLOW
    3: (152, 135, 122), # SHORE
    4: (92, 114, 81),   # GRASS
    5: (120, 119, 112), # MOUNTAIN
    6: (225, 227, 222), # SNOW
    7: (219, 203, 165)  # SAND
}


ENTITY_COLORS = {
    "flora": (67, 124, 23),       # Verde bosque mate
    "fauna": (153, 76, 0),        # Marrón mate
    "human": (55, 65, 120),       # Azul grisáceo mate

    "oak_tree": (25, 80, 20),     # Verde oscuro mate
    "bramble": (95, 117, 57),     # Verde oliva mate
    "mushroom": (140, 82, 45),    # Marrón mate
    "deer": (140, 94, 72),        # Marrón claro mate
    "boar": (119, 69, 19),        # Marrón oscuro mate
    "fox": (180, 80, 45)          # Naranja rojizo mate
}


def find_latest_snapshot(snapshot_dir: str = "simulation_records") -> str:
    snapshot_path = Path(os.path.join(SIMULATION_DIR, snapshot_dir))

    if not snapshot_path.exists() or not snapshot_path.is_dir():
        return ""

    snapshot_files = list(snapshot_path.glob("biome_snapshot_*.json"))

    if not snapshot_files:
        return ""

    snapshot_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return str(snapshot_files[0])

def parse_args():
    parser = argparse.ArgumentParser(description="Echoes of Gaia Snapshot Viewer")

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to the snapshot file (if not specified, the most recent is used)"
    )

    parser.add_argument(
        "-s", "--cell-size",
        type=int,
        default=20,
        help="Cell size in pixels (default: 20)"
    )

    parser.add_argument(
        "-w", "--width",
        type=int,
        default=2000,
        help="Window width (default: 1800)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1300,
        help="Window height (default: 1000)"
    )

    parser.add_argument(
        "-p", "--panel-width",
        type=int,
        default=600,
        help="Information panel width (default: 450)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second (default: 60)"
    )

    parser.add_argument(
        "--font-size",
        type=int,
        default=22,  # Aumentado de 16 a 18
        help="Font size (default: 18)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    LoggerManager.initialize(args.log_level)
    logger = LoggerManager.get_logger("visualization")

    if not args.file:
        latest_snapshot = find_latest_snapshot()
        if not latest_snapshot:
            logger.error("No file specified and no snapshot found in simulation_records/")
            sys.exit(1)

        args.file = latest_snapshot
        logger.info(f"Using the most recent snapshot: {args.file}")

    snapshot_path = Path(args.file)
    if not snapshot_path.exists():
        logger.error(f"The file {args.file} does not exist")
        sys.exit(1)

    config: ViewerConfig = {
        "cell_size": args.cell_size,
        "panel_width": args.panel_width,
        "window_size": (args.width, args.height),
        "fps": args.fps,
        "font_size": args.font_size,
        "title": "Echoes of Gaia - Snapshot Viewer",
        "background_color": (15, 15, 20),
        "grid_color": (40, 60, 80),
        "terrain_colors": TERRAIN_COLORS,
        "entity_colors": ENTITY_COLORS,
        "navigation_button_size": (40, 40),
        "snapshot_path": str(snapshot_path),
        "use_terrain_sprites": False
    }

    viewer = SnapshotViewer(config)
    viewer.run()


if __name__ == "__main__":
    main()
