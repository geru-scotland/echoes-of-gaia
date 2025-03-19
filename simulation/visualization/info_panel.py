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
from typing import Dict, Optional, Tuple, List, Any

import pygame

from simulation.visualization.entity_renderer import EntityInfo
from simulation.visualization.types import SnapshotTimeInfo, MetricsData, BiomeScoreData, TerrainTypeInfo, Color


def format_time_value(ticks: float, timers) -> str:
    days = ticks / timers.Calendar.DAY
    years = int(days / 365)
    remaining_days = int(days % 365)
    months = int(remaining_days / 30)
    days = int(remaining_days % 30)

    return f"{years}a {months}m {days}d"

class InfoPanel:
    def __init__(
            self,
            width: int,
            height: int,
            background_color: Color,
            text_color: Color = (220, 220, 225),
            font_size: int = 30
    ):
        self._logger = logging.getLogger("info_panel")
        self._width = width
        self._height = height
        self._background_color = background_color
        self._text_color = text_color
        self._font_size = font_size

        pygame.font.init()
        self._font = pygame.font.SysFont(None, font_size)
        self._font_bold = pygame.font.SysFont(None, font_size)
        self._font_bold.set_bold(True)
        self._font_title = pygame.font.SysFont(None, font_size + 6)
        self._font_title.set_bold(True)

        self._surface = pygame.Surface((width, height))

        self._simulation_time: Optional[SnapshotTimeInfo] = None
        self._metrics: Optional[MetricsData] = None
        self._biome_score: Optional[BiomeScoreData] = None
        self._selected_entity: Optional[EntityInfo] = None
        self._selected_terrain: Optional[TerrainTypeInfo] = None

        self._quality_colors = {
            "critical": (180, 50, 50),      # Rojo mate
            "unstable": (180, 100, 50),     # Naranja mate
            "moderate": (180, 160, 50),     # Amarillo mate
            "healthy": (100, 160, 50),      # Verde claro mate
            "eden": (50, 160, 80)           # Verde mate
        }

    def _format_field_value(self, field: str, value: Any) -> str:
        time_related_fields = {
            "age", "birth_tick", "biological_age", "lifespan_in_ticks", "lifespan",
            "death_tick"
        }

        if field in time_related_fields and isinstance(value, (int, float)):
            from shared.timers import Timers
            return format_time_value(value, Timers)

        elif isinstance(value, float):
            return f"{value:.2f}"

        elif isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            return "{...}"

        elif isinstance(value, bool):
            return "✓" if value else "✗"

        else:
            return str(value)

    def set_simulation_time(self, simulation_time: SnapshotTimeInfo) -> None:
        self._simulation_time = simulation_time

    def set_metrics(self, metrics: MetricsData) -> None:
        self._metrics = metrics

    def set_biome_score(self, biome_score: BiomeScoreData) -> None:
        self._biome_score = biome_score

    def set_selected_entity(self, entity: Optional[EntityInfo]) -> None:
        self._selected_entity = entity

    def set_selected_terrain(self, terrain: Optional[TerrainTypeInfo]) -> None:
        self._selected_terrain = terrain

    def _render_text_line(self, surface: pygame.Surface, text: str, position: Tuple[int, int],
                          color: Color = None, font: pygame.font.Font = None) -> int:
        if color is None:
            color = self._text_color

        if font is None:
            font = self._font

        text_surface = font.render(text, True, color)
        surface.blit(text_surface, position)

        return position[1] + text_surface.get_height() + 5

    def render(self, surface: pygame.Surface, position: Tuple[int, int] = (0, 0)) -> None:
        self._surface.fill(self._background_color)

        panel_rect = pygame.Rect(0, 0, self._width, self._height)
        pygame.draw.rect(self._surface, (45, 45, 50), panel_rect, 1)
        shadow_surface = pygame.Surface((self._width, 4), pygame.SRCALPHA)
        shadow_surface.fill((0, 0, 0, 40))  # Negro semitransparente
        self._surface.blit(shadow_surface, (0, 0))
        x, y = 10, 10

        def add_section_separator(y_pos):
            separator_rect = pygame.Rect(15, y_pos, self._width - 30, 1)
            pygame.draw.rect(self._surface, (50, 50, 55), separator_rect)
            return y_pos + 15

        if self._simulation_time:
            y = self._render_text_line(self._surface, "TIEMPO DE SIMULACIÓN", (x, y), font=self._font_title)
            y = self._render_text_line(self._surface, f"Año: {self._simulation_time['year']}", (x, y))
            y = self._render_text_line(self._surface, f"Mes: {self._simulation_time['month']}", (x, y))
            y = self._render_text_line(self._surface, f"Ticks: {self._simulation_time['raw_ticks']}", (x, y))
            y = add_section_separator(y + 5)

        if self._metrics:
            y = self._render_text_line(self._surface, "MÉTRICAS DEL BIOMA", (x, y), font=self._font_title)
            y = self._render_text_line(self._surface, f"Flora: {self._metrics['num_flora']}", (x, y))
            y = self._render_text_line(self._surface, f"Fauna: {self._metrics['num_fauna']}", (x, y))

            if 'avg_toxicity' in self._metrics:
                y = self._render_text_line(self._surface, f"Toxicidad media: {self._metrics['avg_toxicity']:.2f}",
                                           (x, y))

            if 'avg_size' in self._metrics:
                y = self._render_text_line(self._surface, f"Tamaño medio: {self._metrics['avg_size']:.2f}", (x, y))

        y = add_section_separator(y + 5)

        if self._biome_score:
            y = self._render_text_line(self._surface, "PUNTUACIÓN DEL BIOMA", (x, y), font=self._font_title)

            quality = self._biome_score["quality"]
            quality_color = self._quality_colors.get(quality, self._text_color)

            y = self._render_text_line(
                self._surface,
                f"Puntuación: {self._biome_score['score']:.2f}/10.0",
                (x, y),
                color=quality_color
            )
            y = self._render_text_line(
                self._surface,
                f"Calidad: {quality.upper()}",
                (x, y),
                color=quality_color,
                font=self._font_bold
            )

            if 'contributor_scores' in self._biome_score:
                y = self._render_text_line(self._surface, "Factores:", (x, y))

                for factor, score in self._biome_score['contributor_scores'].items():
                    factor_name = factor.replace('_', ' ').title()
                    y = self._render_text_line(self._surface, f"  - {factor_name}: {score:.2f}", (x, y))

        y = add_section_separator(y + 5)

        if self._selected_entity:
            y = self._render_text_line(self._surface, "ENTIDAD SELECCIONADA", (x, y), font=self._font_title)
            y = self._render_text_line(self._surface, f"ID: {self._selected_entity.id}", (x, y))
            y = self._render_text_line(self._surface, f"Tipo: {self._selected_entity.type}", (x, y))
            y = self._render_text_line(self._surface, f"Especie: {self._selected_entity.species}", (x, y))
            y = self._render_text_line(self._surface, f"Posición: {self._selected_entity.position}", (x, y))

            if self._selected_entity.habitats:
                habitats_str = ", ".join(self._selected_entity.habitats)
                y = self._render_text_line(self._surface, f"Hábitats: {habitats_str}", (x, y))

            fields_by_component = self._selected_entity.state_fields

            for component_name, fields in fields_by_component.items():
                if not fields:
                    continue

                component_title = component_name.replace('_', ' ').title()
                y = self._render_text_line(self._surface, f"-- {component_title} --", (x, y), font=self._font_bold)

                for field, value in fields.items():
                    if value is None:
                        continue

                    field_name = field.replace('_', ' ').title()

                    formatted_value = self._format_field_value(field, value)

                    y = self._render_text_line(self._surface, f"  {field_name}: {formatted_value}", (x, y))

                y += 15

        elif self._selected_terrain:
            y = self._render_text_line(self._surface, "TERRENO SELECCIONADO", (x, y), font=self._font_title)
            y = self._render_text_line(self._surface, f"ID: {self._selected_terrain['id']}", (x, y))
            y = self._render_text_line(self._surface, f"Tipo: {self._selected_terrain['name']}", (x, y))

            terrain_color_rect = pygame.Rect(x, y, 20, 20)
            pygame.draw.rect(self._surface, self._selected_terrain['color'], terrain_color_rect)
            pygame.draw.rect(self._surface, (255, 255, 255), terrain_color_rect, 1)

            y += 30

        surface.blit(self._surface, position)

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height