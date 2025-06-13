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
Collector system for comprehensive biome state aggregation.

Gathers entity, terrain, climate and metrics data into structured snapshots;
coordinates data extraction from various simulation subsystems.
Integrates with neurosymbolic data services for AI analysis - provides
complete point-in-time capture of simulation state for analysis and persistence.
"""

import time
import traceback
from logging import Logger
from typing import Dict, Any, Optional

import numpy as np

from biome.entities.entity import Entity
from biome.services.climate_service import ClimateService
from biome.systems.managers.climate_data_manager import ClimateDataManager
from biome.systems.managers.entity_manager import EntityProvider
from biome.systems.maps.worldmap import WorldMap
from biome.systems.metrics.analyzers.biome_score import BiomeScoreAnalyzer, BiomeScoreResult
from biome.systems.metrics.analyzers.contributors import ClimateContributor
from biome.systems.metrics.collectors.climate_collector import ClimateDataCollector
from biome.systems.metrics.collectors.entity_collector import EntityDataCollector
from biome.systems.neurosymbolics.data_service import NeurosymbolicDataService
from biome.systems.snapshots.data import SnapshotData
from shared.enums.enums import TerrainType, BiomeType, Season, EntityType, DietType
from shared.enums.strings import Loggers
from shared.types import TerrainData, EntityData, ComponentData, ClimateData, TerrainMap, MetricsData, BiomeScoreData
from simulation.core.systems.time.time import SimulationTimeInfo
from utils.loggers import LoggerManager


class SnapshotCollector:
    def __init__(self, biome_type: BiomeType, entity_manager: EntityProvider, climate_data_manager: ClimateDataManager,
                 world_map: WorldMap,
                 entity_collector: EntityDataCollector, score_analyzer: BiomeScoreAnalyzer,
                 climate_collector: ClimateDataCollector = None, dataset_generation: bool = False):
        self._logger: Logger = LoggerManager.get_logger(Loggers.BIOME)
        self._biome_type: BiomeType = biome_type
        self._entity_manager: EntityProvider = entity_manager
        self._climate_data_manager: ClimateDataManager = climate_data_manager
        self._entity_collector: EntityDataCollector = entity_collector
        self._world_map: WorldMap = world_map
        self._climate_collector: ClimateDataCollector = climate_collector
        self._score_analyzer: BiomeScoreAnalyzer = score_analyzer
        self._dataset_generation: bool = dataset_generation
        self._neurosymbolic_data: Dict[str, Any] = {}

    def collect_snapshot_data(self, simulation_time: int, snapshot_id: int) -> SnapshotData:
        snapshot_time = f"{int(time.time())}_{simulation_time}"
        time_info = SimulationTimeInfo.from_ticks(simulation_time)

        snapshot = SnapshotData(snapshot_time, time_info)
        current_season: Season = ClimateService.get_current_season()
        snapshot.set_biome_info(self._biome_type, current_season)

        snapshot.set_terrain_data(self._collect_terrain_data())

        self._collect_entities_data(snapshot)

        self._collect_metrics_data(snapshot)

        climate_data: ClimateData = self._collect_climate_data()
        snapshot.set_climate_data(climate_data)

        self._collect_climate_analysis_data(snapshot, climate_data)

        self._neurosymbolic_data = self._collect_neurosymbolic_data(snapshot, snapshot_id)

        return snapshot

    def _collect_climate_analysis_data(self, snapshot: SnapshotData, climate_data: ClimateData) -> None:
        current_season: Season = ClimateService.get_current_season()
        biome_data = {
            "biome_type": self._biome_type,
            "current_season": current_season,
            "climate_averages": climate_data
        }

        if snapshot.metrics_data:
            biome_data.update(snapshot.metrics_data)

        climate_contributor = ClimateContributor()
        climate_contributor.calculate(biome_data)

        if "climate_analysis" in biome_data:
            snapshot.climate_analysis = biome_data["climate_analysis"]

    def _collect_climate_data(self) -> ClimateData:
        try:
            climate_averages = self._climate_data_manager.get_current_month_averages()
            climate_system = self._climate_data_manager.get_climate_system()

            climate_averages["co2_level"] = climate_system.get_co2_level()
            climate_averages["biomass_index"] = climate_system.get_biomass_index()
            climate_averages["atmospheric_pressure"] = climate_system.get_atmospheric_pressure()

            current_weather = climate_system.get_current_weather_event()
            if current_weather:
                climate_averages["current_weather"] = str(current_weather)

            return climate_averages
        except Exception as e:
            self._logger.error(f"Error collecting climate data: {e}")
            return {"error": str(e)}

    def _collect_terrain_data(self) -> TerrainData:
        try:
            terrain_map: TerrainMap = self._world_map.terrain_map

            # Serializable, convierto los terrenos a int.
            terrain_int_map = terrain_map.astype(np.int8).tolist()

            terrain_types = {int(terrain): terrain.name for terrain in TerrainType}

            map_shape = terrain_map.shape
            terrain_data = {
                "terrain_map": terrain_int_map,
                "terrain_types": terrain_types,
                "map_dimensions": (map_shape[0] - 1, map_shape[1] - 1)
            }

            return terrain_data
        except Exception as e:
            self._logger.error(f"Error collecting terrain data: {e}")
            return {"error": str(e)}

    def _collect_entities_data(self, snapshot: SnapshotData) -> None:
        try:
            flora, fauna = self._entity_manager.get_entities()
            all_entities = flora + fauna

            for entity in all_entities:
                entity_data = self._collect_entity_data(entity)
                snapshot.add_entity_data(entity.get_id(), entity_data)

        except Exception as e:
            self._logger.error(f"Error collecting entities data: {e}")

    def _collect_entity_data(self, entity: Entity) -> EntityData:
        try:
            entity_data = {
                "id": entity.get_id(),
                "type": entity.get_type(),
                "species": str(entity.get_species()),
                "habitats": [str(h) for h in entity.get_habitats()],
                "state_fields": entity.get_state_fields(),
                "is_dead": entity.get_state_fields().get("general", {}).get("is_dead"),
                "components": self._collect_entity_components(entity),
                "evolution_cycle": entity.get_state_fields().get("general").get("evolution_cycle", 0),
            }

            if entity.get_type() == EntityType.FAUNA:
                if entity.diet_type:
                    entity_data["diet_type"] = str(entity.diet_type.value)

            return entity_data

        except Exception as e:
            self._logger.error(f"Error collecting data for entity {entity.get_id()}: {e}")
            return {"id": entity.get_id(), "error": str(e)}

    def _collect_entity_components(self, entity: Entity) -> Dict[str, ComponentData]:
        components_data = {}
        for component_type, component in entity.components.items():
            try:
                component_attrs = {
                    k: v for k, v in vars(component).items()
                    if not k.startswith('_') and not callable(v)
                }

                components_data[str(component_type)] = component_attrs
            except Exception as e:
                self._logger.error(f"Error collecting component {component_type} data: {e}")

        return components_data

    def _collect_metrics_data(self, snapshot: SnapshotData) -> None:
        try:
            biome_statistics = self._entity_collector.collect_data()

            biome_score_result, contributor_scores = self._score_analyzer.calculate_score(biome_statistics)

            snapshot.set_metrics_data(biome_statistics)
            snapshot.set_biome_score({
                **biome_score_result.to_dict(),
                "contributor_scores": contributor_scores
            })

        except Exception as e:
            self._logger.error(f"Error collecting metrics data: {e}")

    def _collect_neurosymbolic_data(self, snapshot: SnapshotData, snapshot_id: int, save_to_files: bool = True) -> Dict[
        str, Any]:
        try:
            biome_statistics: MetricsData = snapshot.metrics_data
            climate_averages: ClimateData = snapshot.climate_averages
            biome_score_data: BiomeScoreData = snapshot.biome_score
            previous_snapshot = NeurosymbolicDataService.get_instance().get_latest_data().get('neural_data', None)

            # 1. Esta parte, sólo para datos globales; para LSTM
            flora, fauna = self._entity_manager.get_entities(only_alive=True)

            herbivore_count = 0
            carnivore_count = 0
            omnivore_count = 0

            species_health = {}
            species_count = {}
            species_stress = {}

            for entity in fauna:
                diet_type = entity.diet_type
                species_name = str(entity.get_species())

                if diet_type == DietType.HERBIVORE:
                    herbivore_count += 1
                elif diet_type == DietType.CARNIVORE:
                    carnivore_count += 1
                elif diet_type == DietType.OMNIVORE:
                    omnivore_count += 1

                if species_name not in species_count:
                    species_count[species_name] = 0
                    species_health[species_name] = 0
                    species_stress[species_name] = 0

                species_count[species_name] += 1

                vital_data = entity.get_state_fields().get('vital', {})
                vitality = vital_data.get('vitality', 0)
                stress = entity.get_state_fields().get('general', {}).get('stress_level', 0)

                species_health[species_name] += vitality
                species_stress[species_name] += stress

            for species in species_count:
                if species_count[species] > 0:
                    species_health[species] /= species_count[species]
                    species_stress[species] /= species_count[species]

            flora_species_count = {}

            for entity in flora:
                species_name = str(entity.get_species())
                if species_name not in flora_species_count:
                    flora_species_count[species_name] = 0
                flora_species_count[species_name] += 1

            prey_population = herbivore_count
            predator_population = carnivore_count + omnivore_count
            predator_prey_ratio = predator_population / max(1, prey_population)

            total_fauna = herbivore_count + carnivore_count + omnivore_count
            total_entities = total_fauna + len(flora)

            biodiversity = biome_statistics.get('biodiversity_index', 0.0)
            biome_score = biome_score_data.get('score', 0.0)

            # Ojo, que me había liado aquí, si bio es 1 y biomescore 10,
            # el ecosystem stability será 1.
            ecosystem_stability = (biodiversity + (biome_score / 10.0)) / 2.0

            avg_stress = biome_statistics.get('avg_stress', 0.0)
            climate_factor = abs(climate_averages.get('avg_temperature', 20.0) - 22.0) / 50.0
            environmental_pressure = (avg_stress + climate_factor) / 2.0

            if previous_snapshot:
                delta_prey = prey_population - previous_snapshot.get('prey_population', prey_population)
                delta_predator = predator_population - previous_snapshot.get('predator_population', predator_population)
                delta_flora = len(flora) - previous_snapshot.get('flora_count', len(flora))
            else:
                delta_prey = 0
                delta_predator = 0
                delta_flora = 0

            neural_features = {
                'snapshot_id': snapshot_id,
                'timestamp': int(time.time()),
                'simulation_time': self._env.now if hasattr(self, '_env') else 0,
                'prey_population': prey_population,
                'predator_population': predator_population,
                'predator_prey_ratio': predator_prey_ratio,
                'delta_prey_population': delta_prey,
                'delta_predator_population': delta_predator,
                'delta_flora_count': delta_flora,
                'avg_stress': biome_statistics.get('avg_stress', 0.0),
                'biome_score': biome_score_data.get('score', 0.0),
                'biodiversity_index': biome_statistics.get('biodiversity_index', 0.0),
                'herbivore_count': herbivore_count,
                'carnivore_count': carnivore_count,
                'flora_count': len(flora),
                'temperature': climate_averages.get('avg_temperature', 20.0) if climate_averages else 20.0,
                'humidity': climate_averages.get('avg_humidity', 50.0) if climate_averages else 50.0,
                'precipitation': climate_averages.get('avg_precipitation', 30.0) if climate_averages else 30.0,
                'co2_level': climate_averages.get('co2_level', 400.0),
                'biomass_density': climate_averages.get('biomass_index', 0.0),
                'atmospheric_pressure': climate_averages.get('atmospheric_pressure', 1012.0),
                'ecosystem_stability': ecosystem_stability,
                'environmental_pressure': environmental_pressure,
                'total_fauna': total_fauna,
                'total_entities': total_entities,
            }

            # 2. Cojo datos específicos de especies para el grafo
            species_data = {}
            for entity in flora:
                species_name = str(entity.get_species())
                if species_name not in species_data:
                    species_data[species_name] = {
                        'type': 'flora',
                        'population': 0,
                        'biomass': 0,
                        'avg_stress': 0,
                        'avg_vitality': 0,
                        'count': 0
                    }

                entity_data = entity.get_state_fields()
                stress = entity_data.get('general', {}).get('stress_level', 0)
                growth_data = entity_data.get('growth', {})

                species_data[species_name]['population'] += 1
                species_data[species_name]['biomass'] += growth_data.get('current_size', 0)
                species_data[species_name]['avg_stress'] += stress
                species_data[species_name]['avg_vitality'] += entity_data.get('vital', {}).get('vitality', 0)
                species_data[species_name]['count'] += 1

            for entity in fauna:
                species_name = str(entity.get_species())
                diet_type = entity.diet_type.value if hasattr(entity, 'diet_type') else 'unknown'

                if species_name not in species_data:
                    species_data[species_name] = {
                        'type': 'fauna',
                        'diet': diet_type,
                        'population': 0,
                        'biomass': 0,
                        'avg_stress': 0,
                        'avg_energy': 0,
                        'avg_vitality': 0,
                        'count': 0
                    }

                entity_data = entity.get_state_fields()
                stress = entity_data.get('general', {}).get('stress_level', 0)
                growth_data = entity_data.get('growth', {})

                species_data[species_name]['population'] += 1
                species_data[species_name]['biomass'] += growth_data.get('current_size', 0)
                species_data[species_name]['avg_stress'] += stress
                species_data[species_name]['avg_vitality'] += entity_data.get('vital', {}).get('vitality', 0)
                species_data[species_name]['avg_energy'] += entity_data.get('heterotrophic_nutrition', {}).get(
                    'energy_reserves', 0)
                species_data[species_name]['count'] += 1

            for species, data in species_data.items():
                count = max(1, data['count'])
                if 'avg_stress' in data:
                    data['avg_stress'] /= count
                if 'avg_vitality' in data:
                    data['avg_vitality'] /= count
                if 'avg_energy' in data:
                    data['avg_energy'] /= count
                if 'biomass' in data:
                    data['biomass'] /= count

            detailed_data = {
                'species_counts': species_count,
                'species_health': species_health,
                'species_stress': species_stress,
                'flora_species_counts': flora_species_count,
            }

            neurosymbolic_data = {
                'neural_data': neural_features,
                'detailed_data': detailed_data,
                'species_data': species_data
            }

            data_service = NeurosymbolicDataService.get_instance()
            data_service.update_data(neural_features, species_data, save_to_files)

            return neurosymbolic_data

        except Exception as e:
            tb = traceback.format_exc()
            self._logger.exception(f"Error Traceback: {tb}")
            self._logger.error(f"Error collecting neurosymbolic data: {e}")
            return {'error': str(e)}

    def get_neurosymbolic_data(self) -> Optional[Dict[str, Any]]:
        return self._neurosymbolic_data
