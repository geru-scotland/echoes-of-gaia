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
from logging import Logger
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.ndimage import convolve

from shared.enums.enums import Habitats, TerrainType
from shared.enums.strings import Loggers
from shared.stores.biome_store import BiomeStore
from shared.types import HabitatCache, TileMap, Position, BiomeStoreData
from utils.loggers import LoggerManager


class HabitatManager:
    def __init__(self, tile_map: TileMap):
        self._logger: Logger = LoggerManager.get_logger(Loggers.WORLDMAP)
        self._tile_map: TileMap = tile_map
        self._habitat_cache: HabitatCache = self._precompute_habitat_cache(BiomeStore.habitats)
        self._base_habitat_map: Dict[Position, List[Habitats.Type]] = self._create_base_habitat_map()

    def _create_base_habitat_map(self) -> Dict[Position, List[Habitats.Type]]:
        base_map = {}

        for habitat, positions in self._habitat_cache.items():
            for pos in positions:
                position = tuple(pos)
                if position not in base_map:
                    base_map[position] = []
                base_map[position].append(habitat)

        return base_map

    def _precompute_habitat_cache(self, habitat_data: BiomeStoreData) -> HabitatCache:
        # TODO: Recuerda liberar del tipo de habitat, la posición de cache al borrar una entidad
        # para todos los habitats donde corresponda, importante para reutilizar.
        self._logger.info("Precomputing habitat cache...")
        try:
            habitat_positions: HabitatCache = {
                Habitats.Type(habitat): np.array([[0, 0]], dtype=np.int8) for habitat, _ in habitat_data.items()
            }
            # es np, pero con terraintype, para optimizar convierto a int8
            terrain_map: ndarray = self._tile_map.astype(np.int8)

            # para convolución una celda, cuenta sus vecinas y no así misma
            kernel_neighbour: ndarray = np.array([[1, 1, 1],
                                                  [1, 0, 1],
                                                  [1, 1, 1]])

            # expando en +1 todos los lados, para que no tenga que ser justo la casilla adyacente
            kernel_expanded = np.pad(kernel_neighbour, pad_width=1, mode='constant', constant_values=1)

            for habitat, rules in habitat_data.items():
                in_terrains: ndarray = np.array([int(getattr(TerrainType, terrain))
                                                 for terrain in rules.get(Habitats.Relations.IN, {})],
                                                dtype=np.int8)
                nearby_terrains: ndarray = np.array([int(getattr(TerrainType, terrain))
                                                     for terrain in rules.get(Habitats.Relations.NEARBY, {})],
                                                    dtype=np.int8)
                in_mask: ndarray = np.isin(terrain_map, in_terrains).astype(np.int8)
                nearby_mask: ndarray = np.isin(terrain_map, nearby_terrains).astype(np.int8)
                valid_positions: ndarray = np.empty((0, 2), dtype=np.int8)

                if nearby_mask.any():
                    nearby_convolved: ndarray = convolve(nearby_mask.astype(np.int8), kernel_expanded,
                                                         mode="constant", cval=0)
                    # Mecaguen la leche, qué dolor de cabeza me ha dado esto.
                    # Si una celda tiene 2 nearby, 1 & 2 dará 0. Necesito
                    # que la máscara proporcione valor true, 1, para todo aquello
                    # que sea mayor que 1... Lo hago con broadcasting y convierto a 0/1 int8
                    nearby_presence = (nearby_convolved > 0).astype(np.int8)
                    habitat_mask: ndarray = in_mask & nearby_presence
                    valid_positions = np.argwhere(habitat_mask)
                    self._logger.debug(f"Habitat {habitat}. \n {terrain_map}")
                    self._logger.debug(f"Habitat mask: \n {habitat_mask}")
                    self._logger.debug(f"Convolved map: \n {nearby_convolved}")
                    self._logger.debug(f"IN mask: \n {in_mask}")
                    self._logger.debug(f"Valid positions for  {habitat}: \n {valid_positions}")
                elif in_mask.any():
                    valid_positions = np.argwhere(in_mask)

                habitat_positions[Habitats.Type(habitat)] = np.array(valid_positions)

            return habitat_positions

        except Exception as e:
            self._logger.error(f"Error precomputing habitat cache: {e}")

    def get_available_position(self, habitats: List[Habitats.Type]) -> Optional[Position]:
        for habitat in habitats:
            habitat_positions: ndarray = self._habitat_cache.get(habitat, np.array([]))
            if habitat_positions.shape[0] > 0:
                random_index = np.random.randint(0, habitat_positions.shape[0])
                selected_position = tuple(habitat_positions[random_index])
                self._logger.debug(f"Selected position {selected_position} for habitat {habitat}")
                return selected_position

        self._logger.warning(f"No available positions found for habitats: {habitats}")
        return None

    def remove_from_available_habitats(self, position: Position) -> None:
        if position is None:
            return

        for habitat, positions in self._habitat_cache.items():
            filtered_positions = np.array([
                pos for pos in positions
                if not (pos[0] == position[0] and pos[1] == position[1])
            ])

            if filtered_positions.shape[0] != positions.shape[0]:
                self._habitat_cache[habitat] = filtered_positions
                self._logger.debug(f"Removed position {position} from habitat {habitat}")

    def restore_habitat(self, position: Position) -> None:
        if position is None:
            return

        original_habitats = self._base_habitat_map.get(position, [])

        if not original_habitats:
            self._logger.debug(f"Position {position} didn't have any original habitats (might be water etc).")
            return

        for habitat in original_habitats:
            if habitat in self._habitat_cache:
                positions = self._habitat_cache[habitat]

                position_exists = any(np.array_equal(position, pos) for pos in positions)

                if not position_exists:
                    new_positions = np.vstack([positions, np.array([position])]) if positions.shape[
                                                                                        0] > 0 else np.array([position])
                    self._habitat_cache[habitat] = new_positions
                    self._logger.debug(f"Restored position {position} to its original habitat {habitat}")

    def position_available_for_habitat(self, position: Position, habitat: Habitats.Type) -> bool:
        terrain_map = self._tile_map.astype(np.int8)
        if position[0] < 0 or position[0] >= terrain_map.shape[0] or position[1] < 0 or position[1] >= \
                terrain_map.shape[1]:
            return False

        rules = BiomeStore.habitats.get(str(habitat), {})
        in_terrains = [getattr(TerrainType, terrain) for terrain in rules.get(Habitats.Relations.IN, {})]

        terrain_at_position = terrain_map[position]
        return TerrainType(terrain_at_position) in in_terrains

    def count_available_positions(self, habitat: Habitats.Type) -> int:
        positions = self._habitat_cache.get(habitat, np.array([]))
        return positions.shape[0]

    @property
    def habitat_cache(self) -> HabitatCache:
        return self._habitat_cache
