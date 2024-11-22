import logging
import os
from pathlib import Path

from utils.paths import LOGS_DIR


class LogColors:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"


class ColorFormatter(logging.Formatter):
    def __init__(self, name, fmt):
        super().__init__(fmt)
        self.name = name

    def format(self, record):
        color = LogColors.RESET
        if record.levelname == "INFO":
            color = LogColors.GREEN
        elif record.levelname == "WARNING":
            color = LogColors.YELLOW
        elif record.levelname == "ERROR":
            color = LogColors.RED
        elif record.levelname == "DEBUG":
            color = LogColors.CYAN

        # aplico color a todo el mensaje
        formatted_message = super().format(record)
        return f"{color}{formatted_message}{LogColors.RESET}"


def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(f'[{name.upper()}] %(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logs_dir = Path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(LOGS_DIR, log_file)

        # sin colores
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # console, con colores
        color_formatter = ColorFormatter(name, f'[{name.upper()}] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

    return logger
