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
import time
from functools import wraps
from typing import Callable, Any

from shared.strings import Loggers
from utils.loggers import LoggerManager

logger = LoggerManager.get_logger(Loggers.TIME)

def log_execution_time(context: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time: float = time.perf_counter()
            result: Any = func(*args, **kwargs)
            end_time: float = time.perf_counter()
            elapsed_time_ms: float = (end_time - start_time) * 1000
            logger.info(f": {context} executed in {elapsed_time_ms:.2f} ms")
            return result
        return wrapper
    return decorator
