import pytest
from unittest.mock import patch, MagicMock

from biome.api.biome_api import BiomeAPI
from config.settings import Settings
from shared.strings import Strings
from simulation.api.simulation_api import SimulationAPI
from simulation.core.bootstrap.context.context_data import BiomeContextData, SimulationContextData
from simulation.core.engine import SimulationEngine
from simulation.core.bootstrap.bootstrap import Bootstrap
import simpy

@pytest.fixture
def settings():
    return Settings()

@pytest.fixture
def simulation_api(settings):
    return SimulationAPI(settings)

@pytest.fixture
def simulation_engine(settings):
    return SimulationEngine(settings)

# ---------------- UNIT TEST ---------------- #

# Test: API se inicializa correctamente
def test_simulation_api_initialization(simulation_api):
    assert simulation_api._engine is None
    assert isinstance(simulation_api._settings, Settings)

# Test: El engine se inicializa correctamente
def test_simulation_engine_initialization(simulation_engine):
    assert isinstance(simulation_engine._env, simpy.Environment)
    assert hasattr(simulation_engine, "_context")
    assert hasattr(simulation_engine, "_biome_api")
    assert hasattr(simulation_engine, "_logger")

# Test: La simulación, sin errores
@patch.object(SimulationEngine, 'run', return_value=None)
def test_simulation_run(mock_run, simulation_api):
    simulation_api.run()
    mock_run.assert_called_once()


# Test: El contexto del bootstrap se inicializa correctamente con datos reales
@patch.object(Bootstrap, 'get_context')
def test_bootstrap_context(mock_get_context, settings):

    logger = MagicMock()
    mock_map = MagicMock()
    mock_config = MagicMock()
    flora_spawns = MagicMock()
    fauna_spawns = MagicMock()

    # instancias válidas de los contextos
    biome_context_data = BiomeContextData(logger=logger, map=mock_map, config=mock_config,
                                          flora_spawns=flora_spawns, fauna_spawns=fauna_spawns)
    simulation_context_data = SimulationContextData(logger=logger, config=mock_config)

    mock_context = MagicMock()
    mock_context.get.side_effect = lambda key: {
        Strings.BIOME_CONTEXT: biome_context_data,
        Strings.SIMULATION_CONTEXT: simulation_context_data
    }.get(key, None)
    mock_context.logger = logger

    mock_get_context.return_value = mock_context

    engine = SimulationEngine(settings)

    assert engine._context is simulation_context_data
    assert engine._biome_api is not None
    assert engine._logger is logger

    mock_get_context.assert_called_once()

# ------------- INTEGRATION TEST ------------- #

def test_simulation_engine_full_initialization(settings):
    engine = SimulationEngine(settings)  # Sin mocks
    assert engine._context is not None
    assert isinstance(engine._biome_api, BiomeAPI)
    assert engine._logger is not None


# ------------- END-TO-END TEST -------------- #

# Test: step avanza en el tiempo
def test_simulation_step_advance(simulation_engine):
    initial_time = simulation_engine._env.now
    simulation_engine.run()
    assert simulation_engine._env.now > initial_time