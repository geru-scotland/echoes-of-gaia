from typing import Dict, Any

from biome.systems.context.context import Context

class ContextManager:
    def __init__(self):
        print("[ContextManager] Initialising context manager")
        self.context = None

    def build_context(self, **kwargs: Any) -> Context:
        print("[ContextManager] Building context")
        # En un principio, el contexto tendrá:
        # 1) Biome settings (info clima, agentes...estado en general)
        # 2) Mapas
        self.context = Context(**kwargs)
        return self.context



        