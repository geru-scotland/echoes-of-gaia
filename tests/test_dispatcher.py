import pytest
from simulation.core.systems.events.dispatcher import EventDispatcher

@pytest.fixture(autouse=True)
def clear_dispatcher():
    EventDispatcher.clear()

def test_event_dispatcher():
    x: int = 0

    def add(n: int):
        nonlocal x
        x = x + n

    EventDispatcher.register("test_add", add)
    EventDispatcher.dispatch("test_add", 14)

    assert x == 14

def test_multiple_listeners():

    resultado = []

    def listener1(message: str):
        resultado.append(f"L1: {message}")

    def listener2(message: str):
        resultado.append(f"L2: {message}")

    EventDispatcher.register("multi_event", listener1)
    EventDispatcher.register("multi_event", listener2)
    EventDispatcher.dispatch("multi_event", "Event received")

    assert resultado == ["L1: Event received", "L2: Event received"]