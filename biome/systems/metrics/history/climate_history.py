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
Climate history tracking service with time-series data management.

Records and organizes climate measurements over evolutionary cycles;
computes monthly averages and exponential moving averages (EMAs).
Provides filtered climate data access by evolution cycle - supports
current period analytics and historical trend assessment operations.
"""

from logging import Logger
from typing import List, Dict, Any, Optional

import pandas as pd
from pandas import DataFrame

from shared.enums.strings import Loggers
from shared.timers import Timers
from utils.loggers import LoggerManager


class ClimateHistoryService:
    def __init__(self):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        self._climate_data = pd.DataFrame({
            "evo_cycle": pd.Series(dtype="int"),
            "day": pd.Series(dtype="int"),
            "month": pd.Series(dtype="int"),
            "temperature": pd.Series(dtype="float"),
            "humidity": pd.Series(dtype="float"),
            "precipitation": pd.Series(dtype="float"),
            "atm_pressure": pd.Series(dtype="float"),
            "current_season": pd.Series(dtype="object"),
        })

    def _compute_monthly_averages(self, evolution_cycle: int) -> DataFrame:
        try:

            if self._climate_data.empty:
                return pd.DataFrame()

            numeric_data: DataFrame = self._climate_data.select_dtypes(include=["number"])
            filtered_data: DataFrame = numeric_data[numeric_data["evo_cycle"] == evolution_cycle]
            monthly_means: DataFrame = filtered_data.groupby('month').mean().reset_index()
            monthly_averages = monthly_means[["evo_cycle", "month", "temperature", "humidity", "precipitation"]]

            return monthly_averages
        except Exception as e:
            self._logger.exception(f"There was an exception computing the monthly averages: {e}")


    def _compute_emas(self, monthly_averages: DataFrame) -> DataFrame:
        try:
            if monthly_averages.empty:
                return pd.DataFrame()

            ema_data: DataFrame = pd.DataFrame(index=monthly_averages.index)

            for column in ["temperature", "humidity", "precipitation"]:
                ema_data[f"{column}_ema"] = monthly_averages[column].ewm(span=6).mean()
                ema_data[f"{column}_avg"] = monthly_averages[column].mean()

            return ema_data
        except Exception as e:
            self._logger.exception(f"There was an exception computing the monthly averages: {e}")


    def add_daily_data(self, daily_data: Dict[str, Any], evolution_cycle: int, env_tick: int):
        day: int = env_tick // Timers.Calendar.DAY
        # TODO: Preparar estos cálculos en clase con métodos de clase
        # precalculados
        month = day // (Timers.Calendar.MONTH // Timers.Calendar.DAY)

        new_entry: DataFrame = pd.DataFrame([{
            "evo_cycle": evolution_cycle,
            "day": day,
            "month": month,
            "temperature": daily_data.get("temperature", 0),
            "humidity": daily_data.get("humidity", 0),
            "precipitation": daily_data.get("precipitation", 0),
            "atm_pressure": daily_data.get("atm_pressure", 0),
            "current_season": daily_data.get("current_season", 0),
        }])

        self._climate_data = pd.concat([self._climate_data, new_entry], ignore_index=True)

    def get_data_by_evolution_cycle(self, evolution_cycle_int) -> DataFrame:
        monthly_averages: DataFrame = self._compute_monthly_averages(evolution_cycle_int)
        data: DataFrame = self._compute_emas(monthly_averages)
        return data

    def get_current_month_averages(self) -> Dict[str, float]:
        if self._climate_data.empty:
            return {
                "avg_temperature": 0.0,
                "avg_humidity": 0.0,
                "avg_precipitation": 0.0
            }

        if len(self._climate_data) == 0:
            return {
                "avg_temperature": 0.0,
                "avg_humidity": 0.0,
                "avg_precipitation": 0.0
            }

        last_cycle = self._climate_data['evo_cycle'].max()
        current_data = self._climate_data[self._climate_data['evo_cycle'] == last_cycle]

        last_records = current_data.tail(30)

        avg_temperature = last_records['temperature'].mean()
        avg_humidity = last_records['humidity'].mean()
        avg_precipitation = last_records['precipitation'].mean()

        return {
            "avg_temperature": avg_temperature,
            "avg_humidity": avg_humidity,
            "avg_precipitation": avg_precipitation
        }