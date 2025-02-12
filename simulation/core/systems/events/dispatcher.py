from typing import Dict, List, Any


class EventDispatcher:
    _listeners: Dict[str, List[Any]] = {}

    @classmethod
    def clear(cls):
        cls._listeners.clear()

    @classmethod
    def register(cls, event_name: str, callback: Any):
        if event_name not in cls._listeners:
            cls._listeners[event_name] = []
        cls._listeners[event_name].append(callback)

    @classmethod
    def trigger(cls, event_name, *args, **kwargs):
        for callback in cls._listeners.get(event_name, []):
            callback(*args, **kwargs)

