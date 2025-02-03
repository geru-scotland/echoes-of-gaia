from typing import Any, Dict

class Context:
    def __init__(self, **kwargs: Any) -> None:
        print("[Context] Creating context.")
        self.data: Dict[str, Any] = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __repr__(self) -> str:
        return f"Context({self.data})"