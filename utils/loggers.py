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

        formatted_message = super().format(record)
        return f"{color}{formatted_message}{LogColors.RESET}"


class ThreadNameFilter(logging.Filter):
    def filter(self, record):
        record.threadname = "Main thread" if threading.current_thread() == threading.main_thread() else "Secondary thread"
        return True


class LoggerManager:

    _loggers = {}
    _log_level = logging.INFO

    @staticmethod
    def initialize(log_level: str):
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
