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
            font_size: int = 28
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
        self._section_states = {
            "tiempo": True,
            "metricas": True,
            "puntuacion": True,
            "general": True,
            "growth": True,
            "vital": True,
            "metabolic": True,
            "nutritional": True,
            "weather_adaptation": True,
            "transform": True
        }
        self._section_buttons = {}
        self._position = (0, 0)

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
        self._position = position
        self._surface.fill((10, 10, 15))

        panel_rect = pygame.Rect(0, 0, self._width, self._height)
        pygame.draw.rect(self._surface, (50, 100, 130), panel_rect, 1)

        col_width = self._width // 2 - 15
        x_left, y_left = 10, 10
        x_right, y_right = self._width // 2 + 5, 10

        if self._simulation_time:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "TIEMPO DE SIMULACIÓN", x_left, y_left, "tiempo")

            if expanded:
                y_left = self._render_text_line(self._surface,
                                                f"Año: {self._simulation_time['year']}",
                                                (x_left + 10, y_next))
                y_left = self._render_text_line(self._surface,
                                                f"Mes: {self._simulation_time['month']}",
                                                (x_left + 10, y_left))
                y_left = self._render_text_line(self._surface,
                                                f"Ticks: {self._simulation_time['raw_ticks']}",
                                                (x_left + 10, y_left))
            else:
                y_left = y_next

        if self._metrics:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "MÉTRICAS DEL BIOMA", x_left, y_left + 10, "metricas")

            if expanded:
                y_left = self._render_text_line(self._surface,
                                                f"Flora: {self._metrics['num_flora']}",
                                                (x_left + 10, y_next))
                y_left = self._render_text_line(self._surface,
                                                f"Fauna: {self._metrics['num_fauna']}",
                                                (x_left + 10, y_left))

                if 'avg_toxicity' in self._metrics:
                    y_left = self._render_text_line(self._surface,
                                                    f"Toxicidad media: {self._metrics['avg_toxicity']:.2f}",
                                                    (x_left + 10, y_left))

                if 'avg_size' in self._metrics:
                    y_left = self._render_text_line(self._surface,
                                                    f"Tamaño medio: {self._metrics['avg_size']:.2f}",
                                                    (x_left + 10, y_left))
            else:
                y_left = y_next

        if self._biome_score:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "PUNTUACIÓN DEL BIOMA", x_left, y_left + 10, "puntuacion")

            if expanded:
                quality = self._biome_score["quality"]
                quality_color = self._quality_colors.get(quality, self._text_color)

                y_left = self._render_text_line(
                    self._surface,
                    f"Puntuación: {self._biome_score['score']:.2f}/10.0",
                    (x_left + 10, y_next),
                    color=quality_color
                )
                y_left = self._render_text_line(
                    self._surface,
                    f"Calidad: {quality.upper()}",
                    (x_left + 10, y_left),
                    color=quality_color,
                    font=self._font_bold
                )

                if 'contributor_scores' in self._biome_score:
                    y_left = self._render_text_line(self._surface,
                                                    "Factores:",
                                                    (x_left + 10, y_left))

                    for factor, score in self._biome_score['contributor_scores'].items():
                        factor_name = factor.replace('_', ' ').title()
                        y_left = self._render_text_line(self._surface,
                                                        f"  - {factor_name}: {score:.2f}",
                                                        (x_left + 10, y_left))
            else:
                y_left = y_next

        if self._selected_entity:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "ENTIDAD SELECCIONADA", x_right, y_right, "entity")

            if expanded:
                y_right = self._render_text_line(self._surface,
                                                 f"ID: {self._selected_entity.id}",
                                                 (x_right + 10, y_next))
                y_right = self._render_text_line(self._surface,
                                                 f"Tipo: {self._selected_entity.type}",
                                                 (x_right + 10, y_right))
                y_right = self._render_text_line(self._surface,
                                                 f"Especie: {self._selected_entity.species}",
                                                 (x_right + 10, y_right))
                y_right = self._render_text_line(self._surface,
                                                 f"Posición: {self._selected_entity.position}",
                                                 (x_right + 10, y_right))

                if self._selected_entity.habitats:
                    habitats_str = ", ".join(self._selected_entity.habitats)
                    y_right = self._render_text_line(self._surface,
                                                     f"Hábitats: {habitats_str}",
                                                     (x_right + 10, y_right))
            else:
                y_right = y_next

            fields_by_component = self._selected_entity.state_fields

            for component_name, fields in fields_by_component.items():
                if not fields:
                    continue

                component_title = component_name.replace('_', ' ').title()
                section_key = component_name.lower()

                y_next, expanded = self._render_collapsible_section(
                    self._surface, f"-- {component_title} --", x_right, y_right + 10, section_key)

                if expanded:
                    section_height = 0
                    for field, value in fields.items():
                        if value is None:
                            continue
                        section_height += 22

                    section_rect = pygame.Rect(x_right, y_next - 3, col_width, section_height)
                    pygame.draw.rect(self._surface, (20, 30, 45), section_rect)
                    pygame.draw.rect(self._surface, (40, 80, 110), section_rect, 1)

                    for field, value in fields.items():
                        if value is None:
                            continue

                        field_name = field.replace('_', ' ').title()
                        formatted_value = self._format_field_value(field, value)

                        if field.lower() in ['vitality', 'energy_reserves'] and isinstance(value, (int, float)):
                            max_field = None
                            if field.lower() == 'vitality' and 'max_vitality' in fields:
                                max_field = 'max_vitality'
                            elif field.lower() == 'energy_reserves' and 'max_energy_reserves' in fields:
                                max_field = 'max_energy_reserves'

                            if max_field and max_field in fields:
                                max_value = fields[max_field]
                                y_next = self._render_field_bar(
                                    self._surface, field_name, value, max_value,
                                    (x_right + 5, y_next), bar_width=col_width - 15)
                                continue

                        text_color = self._text_color
                        if isinstance(value, (int, float)) and field.lower() in ['vitality', 'health',
                                                                                 'energy_reserves']:
                            max_value = 100.0

                            if field.lower() == 'vitality' and 'max_vitality' in fields:
                                max_value = fields['max_vitality']
                            elif field.lower() == 'energy_reserves' and 'max_energy_reserves' in fields:
                                max_value = fields['max_energy_reserves']

                            percentage = value / max_value if max_value > 0 else 0

                            if percentage < 0.3:
                                text_color = (220, 100, 100)
                            elif percentage > 0.7:
                                text_color = (100, 220, 100)
                            else:
                                text_color = (220, 220, 100)

                        line_text = f"  {field_name}: {formatted_value}"
                        y_next = self._render_text_line(self._surface, line_text,
                                                        (x_right + 5, y_next), color=text_color)

                y_right = y_next if expanded else y_next

        elif self._selected_terrain:
            y_right = self._render_text_line(self._surface,
                                             "TERRENO SELECCIONADO",
                                             (x_right, y_right),
                                             font=self._font_title)
            y_right = self._render_text_line(self._surface,
                                             f"ID: {self._selected_terrain['id']}",
                                             (x_right + 10, y_right))
            y_right = self._render_text_line(self._surface,
                                             f"Tipo: {self._selected_terrain['name']}",
                                             (x_right + 10, y_right))

            terrain_color_rect = pygame.Rect(x_right + 10, y_right, 20, 20)
            pygame.draw.rect(self._surface, self._selected_terrain['color'], terrain_color_rect)
            pygame.draw.rect(self._surface, (255, 255, 255), terrain_color_rect, 1)
            y_right += 30

        mid_x = self._width // 2
        pygame.draw.line(self._surface, (50, 100, 130),
                         (mid_x, 10), (mid_x, self._height - 10), 1)

        surface.blit(self._surface, position)

    def _render_text_with_background(self, surface: pygame.Surface, text: str, position: Tuple[int, int],
                                     color: Color = None, font: pygame.font.Font = None,
                                     bg_color: Color = (25, 35, 45)) -> int:
        if color is None:
            color = self._text_color

        if font is None:
            font = self._font

        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(topleft=position)

        bg_rect = text_rect.inflate(8, 4)
        pygame.draw.rect(surface, bg_color, bg_rect)
        pygame.draw.rect(surface, (40, 80, 100), bg_rect, 1)

        surface.blit(text_surface, text_rect)
        return position[1] + text_surface.get_height() + 5

    def _render_field_bar(self, surface: pygame.Surface, name: str, value: float, max_value: float,
                          position: Tuple[int, int], bar_width: int = 100) -> int:
        x, y = position
        percentage = min(1.0, max(0.0, value / max_value))

        if percentage < 0.3:
            color = (180, 60, 60)
        elif percentage < 0.6:
            color = (180, 180, 60)
        else:
            color = (60, 180, 60)

        font = self._font
        text = f"{name}: {value:.1f}/{max_value:.1f}"
        text_surface = font.render(text, True, self._text_color)
        text_rect = text_surface.get_rect(topleft=(x, y))
        surface.blit(text_surface, text_rect)

        bar_y = y + text_rect.height + 3
        bar_bg_rect = pygame.Rect(x, bar_y, bar_width, 6)
        bar_fg_rect = pygame.Rect(x, bar_y, int(bar_width * percentage), 6)

        pygame.draw.rect(surface, (40, 40, 50), bar_bg_rect)
        pygame.draw.rect(surface, color, bar_fg_rect)
        pygame.draw.rect(surface, (100, 100, 120), bar_bg_rect, 1)

        return bar_y + 10

    def handle_event(self, event, panel_position):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._position = panel_position
            local_pos = (event.pos[0] - panel_position[0], event.pos[1] - panel_position[1])

            for section_key, button_rect in self._section_buttons.items():
                if button_rect.collidepoint(local_pos):
                    self._section_states[section_key] = not self._section_states.get(section_key, True)
                    return True

        return False

    def _render_collapsible_section(self, surface, title, x, y, section_key):
        expanded = self._section_states.get(section_key, True)
        button_text = "[-]" if expanded else "[+]"

        title_font = self._font_title
        title_surface = title_font.render(title, True, (120, 180, 220))
        button_surface = self._font.render(button_text, True, (180, 180, 180))

        surface.blit(title_surface, (x, y))

        button_x = x + title_surface.get_width() + 10
        button_y = y
        button_rect = button_surface.get_rect(topleft=(button_x, button_y))
        surface.blit(button_surface, button_rect)

        self._section_buttons[section_key] = button_rect

        return y + title_surface.get_height() + 5, expanded

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height