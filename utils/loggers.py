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
from typing import Dict, Optional

from utils.paths import LOGS_DIR


class LogColors:
    RESET = "\033[0m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    ORANGE = "\033[38;5;208m"

    DEEP_PURPLE = "\033[38;5;60m"
    SLATE_BLUE = "\033[38;5;68m"
    STEEL_BLUE = "\033[38;5;67m"
    OLIVE_GREEN = "\033[38;5;100m"
    MOSS_GREEN = "\033[38;5;65m"
    BURNT_ORANGE = "\033[38;5;130m"
    COPPER = "\033[38;5;136m"
    DARK_RED = "\033[38;5;88m"
    CHARCOAL = "\033[38;5;240m"
    STONE_GRAY = "\033[38;5;244m"

    LOGGER_COLORS: Dict[str, str] = {
        "bootstrap": CYAN,
        "simulation": ORANGE,
        "render": BRIGHT_BLUE,
        "game": BRIGHT_GREEN,
        "scene": BRIGHT_WHITE,
        "biome": GREEN,
        "research": DEEP_PURPLE,
        "world_map": SLATE_BLUE,
        "time": YELLOW,
        "influxdb": MOSS_GREEN
    }

    LEVEL_COLORS: Dict[str, str] = {
        "DEBUG": BRIGHT_MAGENTA,
        "INFO": GRAY,
        "WARNING": ORANGE,
        "ERROR": BRIGHT_RED,
        "CRITICAL": BRIGHT_RED,
    }


class ColorFormatter(logging.Formatter):
    def __init__(self, name: str, fmt: str):
        super().__init__(fmt)
        self.name = name

    def format(self, record: logging.LogRecord) -> str:
        logger_color = LogColors.LOGGER_COLORS.get(record.name, LogColors.RESET)
        level_color = LogColors.LEVEL_COLORS.get(record.levelname, LogColors.RESET)

        original_msg = record.getMessage()
        original_level = record.levelname
        original_time = self.formatTime(record, self.datefmt)
        threadname = getattr(record, 'threadname', 'Main thread')

        log_line = (
            f"{logger_color}[{record.name.upper()}][{threadname}] {original_time} "
            f"- {record.name} - "
            f"{level_color}{original_level}"
            f"{logger_color} - "
            f"{level_color}{original_msg}"
            f"{LogColors.RESET}"
        )

        return log_line


class ThreadNameFilter(logging.Filter):
    """
    Filtro que añade información detallada del hilo al registro de logs.
    Identifica el hilo principal y asigna nombres únicos a hilos secundarios.
    """

    def __init__(self, thread_prefix: str = "Thread"):
        super().__init__()
        self._thread_prefix = thread_prefix

    def filter(self, record):
        current_thread = threading.current_thread()

        if current_thread == threading.main_thread():
            record.threadname = "Main thread"
        else:
            thread_name = current_thread.name
            thread_id = current_thread.ident

            if thread_name and thread_name != "Thread-" + str(thread_id):
                record.threadname = thread_name
            else:
                record.threadname = f"{self._thread_prefix}-{thread_id}"

        return True

import logging

class LoggerManager:
    _loggers = {}
    _log_level = logging.INFO

    @staticmethod
    def initialize(log_level: str):
        if log_level.upper() == "OFF":
            logging.disable(logging.CRITICAL + 1)  # Desactiva completamente los logs
        else:
            LoggerManager._log_level = getattr(logging, log_level.upper(), logging.INFO)


    @staticmethod
    def get_logger(name: str, log_file: str = None):
        if name in LoggerManager._loggers:
            return LoggerManager._loggers[name]

        log_file = log_file or f"{name}.log"
        logger = logging.getLogger(name)
        logger.setLevel(LoggerManager._log_level)

        if not logger.handlers:
            formatter = logging.Formatter(
                f'[{name.upper()}][%(threadname)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            logs_dir = Path(LOGS_DIR)
            logs_dir.mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(LOGS_DIR, log_file)

            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(ThreadNameFilter())
            logger.addHandler(file_handler)

            color_formatter = ColorFormatter(name,
                                             f'[{name.upper()}][%(threadname)s] %(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(color_formatter)
            console_handler.addFilter(ThreadNameFilter())
            logger.addHandler(console_handler)

        LoggerManager._loggers[name] = logger
        return logger
