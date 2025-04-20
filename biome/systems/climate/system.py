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
import random
from logging import Logger
from typing import Dict, Any, Optional, Callable

from biome.services.climate_service import ClimateService
from biome.systems.climate.seasons import SeasonSystem
from biome.systems.climate.state import ClimateState
from biome.systems.events.event_bus import BiomeEventBus
from biome.systems.managers.entity_manager import EntityProvider
from shared.enums.enums import BiomeType, Season, WeatherEvent, ComponentType
from shared.enums.events import BiomeEvent
from shared.enums.thresholds import ClimateThresholds
from shared.stores.biome_store import BiomeStore
from shared.enums.strings import Loggers
from utils.loggers import LoggerManager
from shared.normalization.normalizer import CLIMATE_RANGES


class ClimateSystem:
    def __init__(self, biome_type: BiomeType, initial_season: Season):
        self._logger: Logger = LoggerManager.get_logger(Loggers.CLIMATE)
        # TODO: Hacer cambio climático - cada x tiempo, forzar una subida de temperatura general, cada 100 años
        # investigar un poco sobre esto.
        self._biome_type: BiomeType = biome_type
        self._entity_provider: Optional[EntityProvider] = None

        self._initial_state: Dict[str, Any] = {}
        self._base_environmental_factors: Dict[str, Any] = {}
        self._weather_event_effects: Dict[WeatherEvent, Dict[Any]] = {}
        self._seasonal_info: Dict[Season, Dict[str, int | float]] = {}
        self._load_environmental_data()

        self._state: ClimateState = self._initialize_state()
        self._season_system: SeasonSystem = SeasonSystem(initial_season)
        ClimateService.init_service(self._state, self._season_system.get_current_season)

        self._record_data_callback: Optional[Callable] = None

    def _load_environmental_data(self):
        self._base_environmental_factors = BiomeStore.biomes.get(self._biome_type, {}).get("environmental_factors", {})
        self._weather_event_effects = BiomeStore.weather_event_effects
        self._seasonal_info = BiomeStore.biomes.get(self._biome_type, {}).get("seasonal_info", {})

    def _initialize_state(self) -> ClimateState:
        try:
            self._initial_state = {
                "temperature": random.randint(
                    self._base_environmental_factors.get("temperature", {}).get("min", 15),
                    self._base_environmental_factors.get("temperature", {}).get("max", 15)
                ),
                "humidity": random.randint(
                    self._base_environmental_factors.get("humidity", {}).get("min", 20),
                    self._base_environmental_factors.get("humidity", {}).get("max", 20)
                ),
                "precipitation": random.randint(
                    self._base_environmental_factors.get("precipitation", {}).get("min", 0),
                    self._base_environmental_factors.get("precipitation", {}).get("max", 0)
                ),
                "biomass_density": self._base_environmental_factors.get("biomass_density", 15),
                "fauna_density": self._base_environmental_factors.get("fauna_density", 15),
                "co2_level": self._base_environmental_factors.get("co2_level", 15),
                "atm_pressure": self._base_environmental_factors.get("atm_pressure", 15),
                "current_weather": None,
            }
            state: ClimateState = ClimateState(**self._initial_state)
            return state
        except Exception as e:
            self._logger.exception(f"An exception occured when trying to obtain environmental factors: {e}")

    def configure_record_callback(self, record_data_callback: Callable):
        self._record_data_callback = record_data_callback

    def update(self, weather_event: WeatherEvent = None) -> None:
        # Este método es el que irá con simpy para el Bioma, lo hago público
        # para que el adaptador de Reinforcement pueda invocarlo
        # Aquí, el modelo debe decidir. Una vez entrenado
        # y en inferencia, llamo al agente para que me de la decision.
        # Pero, para entrenamiento al ser cálculos simulados en un entorno
        # aislado, calculo los factores en el adapter.
        self._season_system.update(self._handle_new_season)
        self._handle_weather_event(weather_event)

    def _handle_new_season(self, season: Season) -> None:
        # por ahora, solo la presión atmosférica cambio
        season_deltas: Dict[str, int | float] = self._seasonal_info.get(season, {}).get("deltas", {})

        if not season_deltas:
            self._logger.error(f"There weren't any seasonal deltas to update the environmental factors.")
            return
        # TODO: Quizá introducir ruido al delta también
        self._state.atm_pressure += season_deltas["atm_pressure"]
        self._logger.debug(
            f"Atmospheric pressure updated: {self._state.atm_pressure} (delta: {season_deltas['atm_pressure']})")

    def _handle_weather_event(self, weather_event: WeatherEvent):
        self._logger.debug(f"Handling weather event: {weather_event}")
        self._logger.debug(f"State previously: {self._state}")

        self._state.current_weather = weather_event

        # TODO: IMPORTANTE. Si el weather event es el mismo que el anterior, reducir en un
        # 0.8 los cambios y que NO se notifique del weather. Si es diferente, NOTIFICAR.
        # TODO: Reentrenar el modelo con este cambio.
        # set_environmental_modifiers del componente metabolico etc. BiomeEventBus y éste que triguee un evento
        # on_weather_change y que las entidades que necesiten saber MODIFICADORES (light, stress etc)
        # que se subscriban. Que utilicen BiomeEventBus.register y los handlers que hagan de callbacks
        # que lo gestionen con el diccionario de componentes.
        min_effect_temp, max_effect_temp = self._weather_event_effects[weather_event]["temperature"].values()
        min_effect_hum, max_effect_hum = self._weather_event_effects[weather_event]["humidity"].values()
        min_effect_prec, max_effect_prec = self._weather_event_effects[weather_event]["precipitation"].values()

        mod_temperature: float = round(random.uniform(min_effect_temp, max_effect_temp), 1)
        mod_precipitation: float = round(random.uniform(min_effect_prec, max_effect_prec), 1)
        mod_humidity: float = round(random.uniform(min_effect_hum, max_effect_hum), 1)

        self._logger.debug(f"Generated values - Temperature: {mod_temperature}")
        self._logger.debug(f"Generated values - Precipitation: {mod_precipitation}")
        self._logger.debug(f"Generated values - Humidity: {mod_humidity}")

        PHYSICAL_MIN_TEMP = CLIMATE_RANGES["temperature"][0]
        PHYSICAL_MAX_TEMP = CLIMATE_RANGES["temperature"][1]
        self._logger.debug(
            f"Temperature - PHYSICAL_MIN: {PHYSICAL_MIN_TEMP}, PHYSICAL_MAX: {PHYSICAL_MAX_TEMP}, "
            f"MOD: {mod_temperature}, CURRENT_STATE: {self._state.temperature}"
        )
        self._state.temperature = round(max(PHYSICAL_MIN_TEMP,
                                            min(self._state.temperature + mod_temperature, PHYSICAL_MAX_TEMP)), 1)

        PHYSICAL_MIN_HUM = CLIMATE_RANGES["humidity"][0]
        PHYSICAL_MAX_HUM = CLIMATE_RANGES["humidity"][1]
        self._logger.debug(
            f"Humidity - PHYSICAL_MIN: {PHYSICAL_MIN_HUM}, PHYSICAL_MAX: {PHYSICAL_MAX_HUM}, "
            f"MOD: {mod_humidity}, CURRENT_STATE: {self._state.humidity}"
        )
        self._state.humidity = round(max(PHYSICAL_MIN_HUM, min(self._state.humidity + mod_humidity, PHYSICAL_MAX_HUM)),
                                     1)

        PHYSICAL_MIN_PREC = CLIMATE_RANGES["precipitation"][0]
        PHYSICAL_MAX_PREC = CLIMATE_RANGES["precipitation"][1]
        self._logger.debug(
            f"Precipitation - PHYSICAL_MIN: {PHYSICAL_MIN_PREC}, PHYSICAL_MAX: {PHYSICAL_MAX_PREC}, "
            f"MOD: {mod_precipitation}, CURRENT_STATE: {self._state.precipitation}"
        )
        self._state.precipitation = round(max(PHYSICAL_MIN_PREC,
                                              min(self._state.precipitation + mod_precipitation, PHYSICAL_MAX_PREC)), 1)

        # TODO: Guardar histórico y actualizar el ClimateState
        # con averages de temp, hum y prec, para que los componentes puedan acceder
        BiomeEventBus.trigger(BiomeEvent.WEATHER_UPDATE,
                              temperature=self._state.temperature,
                              weather_event=weather_event)
        self._logger.debug(
            f"\n{'=' * 60}\n"
            f"[CLIMATE] Season: {self._season_system.get_current_season()} | Weather Event: {weather_event}\n"
            f"│ Temperature: {self._state.temperature:.2f}°C\n"
            f"│ Humidity: {self._state.humidity:.2f}%\n"
            f"│ Precipitation: {self._state.precipitation:.2f}mm\n"
            f"{'=' * 60}"
        )

    def environmental_factors_update(self):
        flora_entities = self._entity_provider.get_flora(only_alive=True)
        fauna_entities = self._entity_provider.get_fauna(only_alive=True)

        total_biomass = 0.0
        total_max_size = 0.0
        total_photosynthesis = 0.0
        total_respiration = 0.0
        total_transpiration = 0.0

        for entity in flora_entities:
            growth_component = entity.get_component(ComponentType.GROWTH)
            if growth_component and growth_component.max_size > 0:
                curr_size = min(growth_component.current_size, growth_component.max_size)
                size_factor = curr_size / growth_component.max_size
                total_biomass += size_factor
                total_max_size += 1.0

            photosynthetic_component = entity.get_component(ComponentType.PHOTOSYNTHETIC_METABOLISM)
            if photosynthetic_component:
                photo_efficiency = photosynthetic_component.photosynthesis_efficiency
                metabolic_activity = photosynthetic_component.metabolic_activity

                if growth_component:
                    size_factor = growth_component.current_size / growth_component.max_size
                    total_photosynthesis += photo_efficiency * metabolic_activity * size_factor * 0.5  # solo cuando hay luz, 50% del dia pongo a ojo de buen cubero

                total_respiration += photosynthetic_component.respiration_rate * size_factor * metabolic_activity * 0.5  # lo mismo, 50% SIN luz, no hay fotosíntesis

                temperature_factor = max(0.0, min(1.0, self._state.temperature / 30.0))
                transpiration_rate = photo_efficiency * metabolic_activity * temperature_factor * 0.1
                if hasattr(photosynthetic_component, 'stress_handler'):
                    stress_ratio = (photosynthetic_component.stress_handler.stress_level
                                    / photosynthetic_component.stress_handler.max_stress)
                    transpiration_rate *= (1.0 - stress_ratio * 0.7)
                if growth_component:
                    transpiration_rate *= size_factor
                total_transpiration += transpiration_rate

        self._state.biomass_density = (
            total_biomass / len(flora_entities)
            if flora_entities else 0.0
        )

        total_fauna_density = 0.0
        total_fauna_entities = 0
        for entity in fauna_entities:
            growth_component = entity.get_component(ComponentType.GROWTH)
            if growth_component and growth_component.max_size > 0:
                curr_size = min(growth_component.current_size, growth_component.max_size)
                size_factor = curr_size / growth_component.max_size
                total_fauna_density += size_factor
                total_fauna_entities += 1

        # densidad media, no suma! pero tengo que revisar esto bien.
        self._state.fauna_density = (
            total_fauna_density / total_fauna_entities
            if total_fauna_entities else 0.0
        )
        total_fauna_size = total_fauna_density

        CO2_EMISSION_PER_FAUNA = 0.8
        CO2_ABSORPTION_PER_BIOMASS = 7.0

        # fauna_co2_flux = total_fauna_size * CO2_EMISSION_PER_FAUNA
        fauna_co2_flux = self._state.fauna_density * CO2_EMISSION_PER_FAUNA
        flora_co2_flux = self._state.biomass_density * CO2_ABSORPTION_PER_BIOMASS
        co2_net_delta = fauna_co2_flux - flora_co2_flux

        new_co2 = self._state.co2_level + co2_net_delta
        self._state.co2_level = max(0.0, min(600.0, new_co2))

        co2_temperature_impact = (self._state.co2_level - 400.0) * 0.01
        co2_temperature_impact = max(-0.1, min(0.1, co2_temperature_impact))
        self._state.temperature += co2_temperature_impact

        humidity_delta = total_transpiration * 3.0
        new_humidity = self._state.humidity + humidity_delta
        self._state.humidity = max(0.0, min(100.0, new_humidity))

        self._logger.debug(
            f"Calculated biomass_density: {self._state.biomass_density:.4f} "
            f"(total_biomass={total_biomass:.4f}, total_entities={len(flora_entities)})"
        )
        self._logger.debug(
            f"Calculated fauna_density: {self._state.fauna_density:.4f}, "
            f"sum_fauna_size={total_fauna_size:.4f}, total_entities={len(fauna_entities)}"
        )
        self._logger.debug(
            f"CO2 level updated to: {self._state.co2_level:.4f} "
            f"with CO2 net delta: {co2_net_delta:.4f} "
            f"(Fauna flux:+{fauna_co2_flux:.2f}, Flora flux:-{flora_co2_flux:.2f})"
        )
        self._logger.debug(
            f"Humidity updated to: {self._state.humidity:.2f}% "
            f"(Transpiration:+{humidity_delta:.2f})"
        )
        self._logger.debug(
            f"Temperature updated to: {self._state.temperature:.4f} "
            f"(CO2 impact:{co2_temperature_impact:.4f})"
        )

    def set_entity_provider(self, entity_provider: EntityProvider) -> None:
        self._entity_provider = entity_provider

    def get_state_and_record(self) -> ClimateState:
        self._record_data_callback()
        return self._state

    def get_state(self) -> ClimateState:
        return self._state

    def get_current_season(self) -> Season:
        return self._season_system.get_current_season()

    def get_seasonal_deltas(self) -> Dict[str, Any]:
        return self._seasonal_info.get(self.get_current_season(), {}).get("deltas", {})

    def get_seasonal_comfort_range(self) -> Dict[str, Any]:
        return self._seasonal_info.get(self.get_current_season(), {}).get("comfort_range", {})

    def get_current_weather_event(self) -> Optional[WeatherEvent]:
        return self._state.current_weather

    def get_co2_level(self) -> float:
        return self._state.co2_level

    def get_biomass_index(self) -> float:
        return self._state.biomass_density

    def get_atmospheric_pressure(self) -> float:
        return self._state.atm_pressure

    @property
    def temperature(self) -> float:
        return self._state.temperature

    @property
    def base_environmental_factors(self) -> Dict[str, int | float]:
        return self._base_environmental_factors

    @property
    def weather_event_effects(self) -> Dict[WeatherEvent, dict]:
        return self._weather_event_effects

    @property
    def seasonal_info(self) -> Dict[Season, Dict[str, int | float]]:
        return self._seasonal_info

    @property
    def seasonal_detals(self) -> Dict[Season, Dict[str, int | float]]:
        return self._seasonal_info

    @property
    def biome_type(self):
        return self._biome_type

    @property
    def data_manager(self):
        return self._record_data_callback
