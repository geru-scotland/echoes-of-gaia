""" 
# =============================================================================
#                                                                              #
#                              ✦ ECHOES OF GAIA ✦                              #
#                                                                              #
#    Trabajo Fin de Grado (TFG)                                                #
#    Facultad de Ingeniería Informática - Donostia                             #
#    UPV/EHU - Euskal Herriko Unibertsitatea                                   #
#                                                                              #
#    Área de Computación e Inteligencia Artificial                             #
#                                                                              #
#    Autor:  Aingeru García Blas                                               #
#    GitHub: https://github.com/geru-scotland                                  #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia                   #
#                                                                              #
# =============================================================================
"""

"""
Asynchronous InfluxDB client for simulation telemetry storage.

Manages database connections and queued write operations;
processes datapoints through dedicated worker thread architecture.
Handles telemetry persistence with proper resource cleanup - ensures
reliable data storage for monitoring and analysis purposes.
"""

from logging import Logger
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict

from influxdb_client import InfluxDBClient, Point, WriteApi, WritePrecision
from influxdb_client.client.write_api import ASYNCHRONOUS

from shared.enums.strings import Loggers
from simulation.core.systems.telemetry.datapoint import Datapoint
from utils.loggers import LoggerManager


class InfluxDB:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._logger: Logger = LoggerManager.get_logger(Loggers.INFLUXDB)
        self._logger.info("Starting InfluxDB logger service")
        self._client: InfluxDBClient = self._connect(config)
        self._bucket: str = config.get("bucket", "echoes_of_gaia")
        self._org: str = config.get("org", "BasajaunOrg")
        self._write_api: WriteApi = self._client.write_api(write_options=ASYNCHRONOUS)
        self._queue: Queue = Queue()
        self._worker_thread = None
        self.running: bool = True

    def _connect(self, config: Dict[str, Any]) -> InfluxDBClient:
        self._logger.info("Connecting to database...")
        try:
            return InfluxDBClient(
                url=config.get("url"),
                token=config.get("token"),
                org=config.get("org")
            )
        except Exception as e:
            self._logger.exception(f"There was an error connecting to InfluxDB: {e}")

    def listen(self) -> None:
        self._logger.info("InfluxDB worker process listening for writing events.")
        self._worker_thread = Thread(target=self._worker, name="InfluxWorkerThread")
        self._worker_thread.start()

    def _worker(self):
        self._logger.info("Worker thread created")
        while True:
            try:
                point = self._queue.get(timeout=1)
                if point is None:
                    self._logger.info("Sentinel sentinel received, closing worker thred.")
                    break
                queue_size = self._queue.qsize()
                self._logger.debug(f"Escribiendo punto: {point}. Tamaño de cola: {queue_size} -> {self._queue.qsize()}")
                try:
                    self._write_api.write(bucket=self._bucket, org=self._org, record=point,
                                          write_precision=WritePrecision.MS)
                except Exception as e:
                    self._logger.error(f"Error al escribir: {e}")
                finally:
                    self._queue.task_done()
            except Empty:
                # si la cola está vacía y el sim engine dice se debe cerrar, terminamos
                if not self.running:
                    break
            except Exception as e:
                self._logger.error(f"There was an error in the worker thread: {e}")

    def write_data_point(self, datapoint: Datapoint):
        if not datapoint.fields:
            self._logger.warning("No fields in datapoint, skipping write.")
            return

        self._logger.debug(
            f"Attempting to write: {datapoint.measurement}, Fields: {datapoint.fields}"
        )
        self._logger.debug(
            f"Enqueuing datapoint: {datapoint.measurement}, Fields: {datapoint.fields}, Queue Size: {self._queue.qsize()}"
        )
        point = Point(datapoint.measurement)

        for tag, value in datapoint.tags.items():
            point.tag(tag, value)

        for field, value in datapoint.fields.items():
            point.field(field, value)

        if datapoint.timestamp:
            point.time(datapoint.timestamp, WritePrecision.MS)

        self._queue.put(point)

    def close(self):
        self._logger.debug("Closing resources...")
        self.running = False
        self._queue.put(None)
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._client.close()