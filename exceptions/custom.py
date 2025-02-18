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

class EchoesOfGaiaException(Exception):
    pass

class SimulationError(EchoesOfGaiaException):
    pass

class BiomeError(EchoesOfGaiaException):
    pass

class DataLoadingError(EchoesOfGaiaException):
    pass

class AgentError(EchoesOfGaiaException):
    pass

class BootstrapError(SimulationError):
    def __init__(self, message: str):
        super().__init__(f"Bootstrap Error: {message}")


class ContextError(SimulationError):
    def __init__(self, message: str):
        super().__init__(f"Context Error: {message}")


class BiomeAPIError(SimulationError):
    def __init__(self, message: str):
        super().__init__(f"BiomeAPI Error: {message}")


class SimulationRunError(SimulationError):
    def __init__(self, message: str):
        super().__init__(f"Simulation run Error: {message}")

class MapGenerationError(BiomeError):
    def __init__(self, message: str):
        super().__init__(f"Map generation error: {message}")


