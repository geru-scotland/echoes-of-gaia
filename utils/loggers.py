import logging
import os
from pathlib import Path

from utils.paths import LOGS_DIR


def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(f'[{name.upper()}] %(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logs_dir = Path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(LOGS_DIR, log_file)

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
