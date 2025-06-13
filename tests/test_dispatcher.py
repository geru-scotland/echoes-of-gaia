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
import pytest

from simulation.core.systems.events.event_bus import SimulationEventBus


@pytest.fixture(autouse=True)
def clear_dispatcher():
    SimulationEventBus.clear()

def test_event_dispatcher():
    x: int = 0

    def add(n: int):
        nonlocal x
        x = x + n

    SimulationEventBus.register("test_add", add)
    SimulationEventBus.trigger("test_add", 14)

    assert x == 14

def test_multiple_listeners():

    resultado = []

    def listener1(message: str):
        resultado.append(f"L1: {message}")

    def listener2(message: str):
        resultado.append(f"L2: {message}")

    SimulationEventBus.register("multi_event", listener1)
    SimulationEventBus.register("multi_event", listener2)
    SimulationEventBus.trigger("multi_event", "Event received")

    assert resultado == ["L1: Event received", "L2: Event received"]