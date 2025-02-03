from typing import Any

from utils.loggers import setup_logger
from biome.systems.maps.procedural_maps import MapGenerator, Map
from biome.systems.context.manager import ContextManager, Context

class BiomeBuilder:
    """
    Cargar settings, inicializar sistemas de eventosGra
    """
    def __init__(self, **kwargs: Any):
        print("[Biome Builder] Initialising BiomeBuilder...")
        self.context_manager: ContextManager = ContextManager()

    def build(self) -> Context:
        print("[Biome Builder] Building biome...")
        # Logs, settings, maps, init events
        map: Map = MapGenerator().generate()
        # build_context ha de recibir settings + map info
        context: Context = self.context_manager.build_context(map=map)
        # devolveré una tupla con logger y contexto
        return context