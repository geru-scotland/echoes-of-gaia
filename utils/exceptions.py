class SimulationError(Exception):
    """Excepción base para todos los errores de la simulación."""
    pass

class BiomeError(Exception):
    """Excepción base para todos los errores de el Bioma"""
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