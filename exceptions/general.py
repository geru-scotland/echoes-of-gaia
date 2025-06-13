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
import logging

from exceptions.custom import EchoesOfGaiaException


def global_exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, EchoesOfGaiaException):
        logging.error(f"[Handled exception]: {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
    else:
        logging.critical(f"[Critical exception]: {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
        # sys.exit(1)

