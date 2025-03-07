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
import logging
from typing import Dict, List, Tuple, Optional, Set

import pygame

from simulation.visualization.types import EntityData, Color, Point


class EntityInfo:
    def __init__(self, entity_data: EntityData, position: Tuple[int, int], color: Color):
        self.id = entity_data["id"]
        self.type = entity_data["type"]
        self.species = entity_data["species"]
        self.position = position
        self.color = color
        self.state_fields = entity_data["state_fields"]
        self.habitats = entity_data["habitats"]

    def __str__(self) -> str:
        return f"Entity {self.id}: {self.species} at {self.position}"


class EntityRenderer:
    def __init__(self, cell_size: int, entity_colors: Dict[str, Color]):
        self._logger = logging.getLogger("entity_renderer")
        self._cell_size = cell_size
        self._entity_colors = entity_colors
        self._entities: Dict[int, EntityInfo] = {}
        self._selected_entity: Optional[int] = None

    def set_entities_data(self, entities_data: Dict[str, EntityData]) -> None:
        try:
            self._entities.clear()

            for entity_id, entity_data in entities_data.items():
                if "components" in entity_data and "transform" in entity_data["components"]:
                    transform = entity_data["components"]["transform"]
                    if "x" in transform and "y" in transform:
                        position = (int(transform["x"]), int(transform["y"]))

                        entity_type = entity_data["type"]
                        specific_type = entity_data["specific_type"]

                        color = self._entity_colors.get(
                            specific_type,
                            self._entity_colors.get(entity_type, (255, 255, 255))
                        )

                        self._entities[int(entity_id)] = EntityInfo(
                            entity_data,
                            position,
                            color
                        )

            self._logger.info(f"Loaded {len(self._entities)} entities")
        except Exception as e:
            self._logger.error(f"Error setting entities data: {e}")

    def render(self, surface: pygame.Surface, offset: Point = (0, 0)) -> None:
        if not self._entities:
            return

        entities_by_type = {"flora": [], "fauna": [], "human": []}
        for entity_id, entity_info in self._entities.items():
            if entity_info.type in entities_by_type:
                entities_by_type[entity_info.type].append(entity_info)
            else:
                entities_by_type.setdefault("other", []).append(entity_info)

        for entity_type in ["flora", "fauna", "human", "other"]:
            if entity_type not in entities_by_type:
                continue

            for entity_info in entities_by_type[entity_type]:
                y, x = entity_info.position

                pixel_x = x * self._cell_size + offset[0] + self._cell_size // 2
                pixel_y = y * self._cell_size + offset[1] + self._cell_size // 2

                radius = self._cell_size // 3 if entity_info.type == "flora" else self._cell_size // 4

                pygame.draw.circle(
                    surface,
                    entity_info.color,
                    (pixel_x, pixel_y),
                    radius,
                    0
                )
                border_color = tuple(max(0, c - 30) for c in entity_info.color)
                pygame.draw.circle(
                    surface,
                    border_color,
                    (pixel_x, pixel_y),
                    radius,
                    1
                )
                if self._selected_entity == entity_info.id:
                    glow_radius = radius + 4
                    glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(
                        glow_surface,
                        (255, 255, 255, 80),
                        (glow_radius, glow_radius),
                        glow_radius
                    )
                    surface.blit(
                        glow_surface,
                        (pixel_x - glow_radius, pixel_y - glow_radius)
                    )

                    pygame.draw.circle(
                        surface,
                        (230, 230, 250),
                        (pixel_x, pixel_y),
                        radius + 2,
                        2
                    )

    def get_entity_at_pos(self, pos: Point, cell_coords: Tuple[int, int]) -> Optional[int]:
        entities_at_cell = []

        for entity_id, entity_info in self._entities.items():
            if entity_info.position == cell_coords:
                entities_at_cell.append((entity_id, entity_info))

        if not entities_at_cell:
            return None

        if len(entities_at_cell) > 1:
            y, x = cell_coords
            cell_center_x = x * self._cell_size + self._cell_size // 2
            cell_center_y = y * self._cell_size + self._cell_size // 2

            closest_entity = None
            min_dist = float('inf')

            for entity_id, entity_info in entities_at_cell:
                dist = ((pos[0] - cell_center_x) ** 2 + (pos[1] - cell_center_y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_entity = entity_id

            return closest_entity

        return entities_at_cell[0][0]

    def select_entity(self, entity_id: Optional[int]) -> None:
        if entity_id is not None and entity_id not in self._entities:
            self._logger.warning(f"Entity {entity_id} not found")
            return

        self._selected_entity = entity_id

    def get_selected_entity(self) -> Optional[int]:
        return self._selected_entity

    def get_entity_info(self, entity_id: int) -> Optional[EntityInfo]:
        return self._entities.get(entity_id)