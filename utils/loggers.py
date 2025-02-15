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
import logging
import os
import threading
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

        if record.name == "bootstrap":
            color = LogColors.CYAN
        elif record.levelname == "INFO":
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

class ThreadNameFilter(logging.Filter):
    def filter(self, record):
        record.threadname = "Main thread" if threading.current_thread() == threading.main_thread() else "Secondary thread"
        return True

def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(f'[{name.upper()}][%(threadname)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logs_dir = Path(LOGS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(LOGS_DIR, log_file)

        # sin colores
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(ThreadNameFilter())
        logger.addHandler(file_handler)

        # console, con colores
        color_formatter = ColorFormatter(name, f'[{name.upper()}][%(threadname)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(color_formatter)
        console_handler.addFilter(ThreadNameFilter())
        logger.addHandler(console_handler)

    return logger
