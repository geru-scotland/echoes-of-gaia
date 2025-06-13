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
import importlib.util

import pytest

if importlib.util.find_spec("stable_baselines3") is None:
    pytest.skip("Skipping test_simulation.py because stable_baselines3 is not installed", allow_module_level=True)


from unittest.mock import MagicMock, patch

import simpy

from biome.api.biome_api import BiomeAPI
from config.settings import Settings
from shared.enums.strings import Loggers, Strings
from simulation.api.simulation_api import SimulationAPI
from simulation.core.bootstrap.bootstrap import Bootstrap
from simulation.core.bootstrap.context.context_data import (
    BiomeContextData,
    SimulationContextData,
)
from simulation.core.engine import SimulationEngine
from utils.loggers import LoggerManager


@pytest.fixture
def settings():
    settings: Settings = Settings()
    LoggerManager.initialize(settings.log_level)
    return settings

@pytest.fixture
def simulation_api(settings):
    return SimulationAPI(settings)

@pytest.fixture
def simulation_engine(settings):
    return SimulationEngine(settings)

@pytest.fixture
def mock_influxdb():
    with patch('simulation.core.systems.telemetry.manager.InfluxDBManager') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


# ---------------- UNIT TEST ---------------- #

# Test: API se inicializa correctamente
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_simulation_api_initialization(simulation_api):
    assert simulation_api._engine is None
    assert isinstance(simulation_api._settings, Settings)

# Test: El engine se inicializa correctamente
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_simulation_engine_initialization(simulation_engine):
    assert isinstance(simulation_engine._env, simpy.Environment)
    assert hasattr(simulation_engine, "_context")
    assert hasattr(simulation_engine, "_biome_api")
    assert hasattr(simulation_engine, "_logger")

# Test: La simulación, sin errores
@patch.object(SimulationEngine, 'run', return_value=None)
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_simulation_run(mock_run, simulation_api):
    simulation_api.run()
    mock_run.assert_called_once()


# Test: El contexto del bootstrap se inicializa correctamente con datos reales
@patch.object(Bootstrap, 'get_context')
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_bootstrap_context(mock_get_context, settings):
    mock_map = MagicMock()
    mock_config = MagicMock()
    flora_spawns = MagicMock()
    fauna_spawns = MagicMock()
    influxdb = MagicMock()

    # instancias válidas de los contextos
    biome_context_data = BiomeContextData(logger_name=Loggers.BIOME, tile_map=mock_map, config=mock_config,
                                          flora_definitions=flora_spawns, fauna_definitions=fauna_spawns)
    simulation_context_data = SimulationContextData(logger_name=Loggers.SIMULATION, config=mock_config, influxdb=influxdb)

    mock_context = MagicMock()
    mock_context.get.side_effect = lambda key: {
        Strings.BIOME_CONTEXT: biome_context_data,
        Strings.SIMULATION_CONTEXT: simulation_context_data
    }.get(key, None)
    logger = LoggerManager.get_logger(Loggers.SIMULATION)
    mock_context.logger = LoggerManager.get_logger(Loggers.SIMULATION)

    mock_get_context.return_value = mock_context

    engine = SimulationEngine(settings)

    assert engine._context is simulation_context_data
    assert engine._biome_api is not None
    assert engine._logger is logger

    mock_get_context.assert_called_once()

# ------------- INTEGRATION TEST ------------- #
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_simulation_engine_full_initialization(settings, mock_influxdb):
    engine = SimulationEngine(settings)  # Sin mocks
    assert engine._context is not None
    assert isinstance(engine._biome_api, BiomeAPI)
    assert engine._logger is not None


# ------------- END-TO-END TEST -------------- #

# Test: step avanza en el tiempo
@pytest.mark.skip(reason="InfluxDB configuration issues")
def test_simulation_step_advance(simulation_engine):
    initial_time = simulation_engine._env.now
    simulation_engine.run()
    assert simulation_engine._env.now > initial_time