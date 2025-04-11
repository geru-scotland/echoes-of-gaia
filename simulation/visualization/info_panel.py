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
        self._font_title = pygame.font.SysFont(None, font_size + 5)
        self._font_title.set_bold(True)

        self._surface = pygame.Surface((width, height))

        self._simulation_time: Optional[SnapshotTimeInfo] = None
        self._metrics: Optional[MetricsData] = None
        self._biome_score: Optional[BiomeScoreData] = None
        self._selected_entity: Optional[EntityInfo] = None
        self._selected_terrain: Optional[TerrainTypeInfo] = None

        self._biome_type = None
        self._current_season = None
        self._climate_averages = None
        self._climate_analysis = None

        self._quality_colors = {
            "critical": (180, 50, 50),  # Rojo mate
            "unstable": (180, 100, 50),  # Naranja mate
            "moderate": (180, 160, 50),  # Amarillo mate
            "healthy": (100, 160, 50),  # Verde claro mate
            "eden": (50, 160, 80)  # Verde mate
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
            "transform": False
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

    def set_biome_type(self, biome_type: str) -> None:
        self._biome_type = biome_type

    def set_current_season(self, season: str) -> None:
        self._current_season = season

    def set_climate_averages(self, averages: Dict[str, float]) -> None:
        self._climate_averages = averages

    def set_climate_analysis(self, analysis: Dict[str, Any]) -> None:
        self._climate_analysis = analysis

    def _render_text_line(self, surface: pygame.Surface, text: str, position: Tuple[int, int],
                          color: Color = None, font: pygame.font.Font = None) -> int:
        if color is None:
            color = self._text_color

        if font is None:
            font = self._font

        text_surface = font.render(text, True, color)
        surface.blit(text_surface, position)

        return position[1] + text_surface.get_height() + 6

    def _render_info_section(self, surface, title, data_items, x, y, width):

        if not data_items:
            return y

        item_height = 28
        section_height = len(data_items) * item_height + 10

        section_rect = pygame.Rect(x - 5, y, width + 10, section_height)
        pygame.draw.rect(surface, (20, 30, 45), section_rect)
        pygame.draw.rect(surface, (40, 70, 110), section_rect, 1)

        header_rect = pygame.Rect(x - 5, y, width + 10, 3)
        pygame.draw.rect(surface, (60, 120, 180), header_rect)

        current_y = y + 8

        for i, (label, value, color) in enumerate(data_items):
            is_category = value == ""

            if is_category:
                row_bg_color = (30, 45, 65)
            else:
                row_bg_color = (25, 35, 50) if i % 2 == 0 else (30, 40, 55)

            row_rect = pygame.Rect(x, current_y - 2, width, item_height)
            pygame.draw.rect(surface, row_bg_color, row_rect)

            if is_category:
                label_surface = self._font_bold.render(label, True, (180, 220, 250))
                surface.blit(label_surface, (x + 5, current_y))

                line_rect = pygame.Rect(x + label_surface.get_width() + 10, current_y + item_height // 2,
                                        width - label_surface.get_width() - 15, 1)
                pygame.draw.rect(surface, (60, 100, 150), line_rect)
            else:
                label_surface = self._font.render(label, True, (180, 200, 220))
                surface.blit(label_surface, (x + 8, current_y))

                if value:
                    max_value_width = width - label_surface.get_width() - 25
                    value_surface = self._font.render(value, True, color)

                    if value_surface.get_width() > max_value_width:
                        smaller_font = pygame.font.SysFont(None, self._font_size - 2)
                        value_surface = smaller_font.render(value, True, color)

                        if value_surface.get_width() > max_value_width:
                            for j in range(len(value) - 3, 0, -1):
                                truncated = value[:j] + "..."
                                value_surface = smaller_font.render(truncated, True, color)
                                if value_surface.get_width() <= max_value_width:
                                    break

                    value_x = x + width - value_surface.get_width() - 10
                    surface.blit(value_surface, (value_x, current_y))

            current_y += item_height

        return current_y + 5

    def _render_climate_analysis(self, surface, analysis_data, x, y, width):

        if not analysis_data:
            return y

        title_height = 30
        factor_height = 30

        factors = analysis_data.get("factors", {})

        factors_height = len(factors) * factor_height
        section_height = title_height + factors_height + 20

        section_rect = pygame.Rect(x - 5, y, width + 10, section_height)
        pygame.draw.rect(surface, (20, 30, 45), section_rect)
        pygame.draw.rect(surface, (40, 70, 110), section_rect, 1)

        current_y = y + 10

        title_text = "Climate Analysis"
        title_surface = self._font_bold.render(title_text, True, (180, 220, 240))
        title_rect = title_surface.get_rect(centerx=x + width // 2, y=current_y)
        self._surface.blit(title_surface, title_rect)
        current_y += title_height

        if "factors" in analysis_data:
            for factor, details in analysis_data["factors"].items():
                factor_name = factor.title()
                value = details["value"]
                status = details["status"]

                if status == "optimal":
                    status_color = (100, 200, 100)
                    status_text = "Optimal"
                elif status == "too_low":
                    status_color = (200, 200, 100)
                    status_text = "Too Low"
                else:
                    status_color = (200, 100, 100)
                    status_text = "Too High"

                factor_text = f"{factor_name}: {value:.1f}"
                factor_surface = self._font.render(factor_text, True, (180, 200, 220))
                self._surface.blit(factor_surface, (x + 10, current_y))

                status_surface = self._font.render(status_text, True, status_color)
                status_x = x + width - status_surface.get_width() - 10
                self._surface.blit(status_surface, (status_x, current_y))

                current_y += factor_height

        return current_y + 10

    def render(self, surface: pygame.Surface, position: Tuple[int, int] = (0, 0)) -> None:
        self._position = position

        self._surface.fill((10, 10, 15))
        is_dead = False
        if self._selected_entity and "general" in self._selected_entity.state_fields:
            is_dead = self._selected_entity.state_fields["general"].get("is_dead", False)

        self._surface.fill((10, 10, 15))

        if is_dead:
            overlay = pygame.Surface((self._width, self._height), pygame.SRCALPHA)
            overlay.fill((120, 30, 30, 40))
            self._surface.blit(overlay, (0, 0))
        panel_rect = pygame.Rect(0, 0, self._width, self._height)
        pygame.draw.rect(self._surface, (50, 100, 130), panel_rect, 1)

        col_width = self._width // 2 - 15
        x_left, y_left = 10, 10
        x_right, y_right = self._width // 2 + 5, 10

        if self._simulation_time:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "SIMULATION TIME", x_left, y_left, "tiempo")

            if expanded:
                ticks = self._simulation_time['raw_ticks']
                from shared.timers import Timers

                months_passed = ticks // Timers.Calendar.MONTH
                year = months_passed // 12
                month = (months_passed % 12) + 1

                days_in_month = 30
                ticks_in_current_month = ticks % Timers.Calendar.MONTH
                day = (ticks_in_current_month // Timers.Calendar.DAY) + 1

                if day > days_in_month:
                    day = days_in_month

                data_items = [
                    ("Date:", f"Year {year}, Month {month}, Day {day}", (180, 220, 230)),
                    ("System Tick:", f"{ticks}", (150, 190, 220))
                ]
                y_left = self._render_info_section(self._surface, "Time", data_items, x_left + 5, y_next,
                                                   col_width - 10)
            else:
                y_left = y_next

        if hasattr(self, '_biome_type') and (
                self._biome_type or self._current_season or self._climate_averages or self._metrics):
            y_next, expanded = self._render_collapsible_section(
                self._surface, "BIOME INFORMATION", x_left, y_left + 10, "biome_info")

            if expanded:
                data_items = []

                if self._biome_type:
                    data_items.append(("Type:", f"{str(self._biome_type).capitalize()}", (160, 220, 180)))

                if self._current_season:
                    season_colors = {
                        "SPRING": (120, 220, 100),
                        "SUMMER": (220, 200, 80),
                        "AUTUMN": (220, 140, 80),
                        "WINTER": (160, 200, 220)
                    }
                    season_str = str(self._current_season)
                    color = season_colors.get(season_str, self._text_color)

                    if "current_weather" in self._climate_averages:
                        weather = self._climate_averages["current_weather"]
                        weather_text = weather.replace("_", " ").title()

                        weather_color = (180, 180, 220)
                        if "rain" in weather.lower() or "storm" in weather.lower():
                            weather_color = (100, 150, 220)
                        elif "sun" in weather.lower() or "clear" in weather.lower():
                            weather_color = (220, 180, 100)
                        elif "snow" in weather.lower() or "blizzard" in weather.lower():
                            weather_color = (220, 220, 250)

                        data_items.append(("Weather:", weather_text, weather_color))

                    data_items.append(("Season:", season_str.capitalize(), color))

                if self._climate_averages:
                    data_items.append(("Climate:", "", (180, 200, 220)))

                    if "avg_temperature" in self._climate_averages:
                        temp = self._climate_averages["avg_temperature"]
                        temp_color = (100, 180, 220)  # Frío
                        if temp > 30:
                            temp_color = (220, 120, 80)  # Caliente
                        elif temp > 15:
                            temp_color = (200, 200, 100)  # Templado
                        data_items.append(("  Temperature:", f"{temp:.1f}°C", temp_color))

                    if "avg_humidity" in self._climate_averages:
                        humidity = self._climate_averages["avg_humidity"]
                        data_items.append(("  Humidity:", f"{humidity:.1f}%", (160, 200, 240)))

                    if "avg_precipitation" in self._climate_averages:
                        precip = self._climate_averages["avg_precipitation"]
                        data_items.append(("  Precipitation:", f"{precip:.1f}mm", (150, 200, 220)))

                    if "co2_level" in self._climate_averages:
                        co2 = self._climate_averages["co2_level"]
                        co2_color = (180, 180, 180)
                        if co2 > 450:
                            co2_color = (220, 120, 80)
                        elif co2 < 380:
                            co2_color = (100, 180, 100)
                        data_items.append(("  CO2 Level:", f"{co2:.1f} ppm", co2_color))

                    if "biomass_index" in self._climate_averages:
                        biomass = self._climate_averages["biomass_index"]
                        biomass_color = (100, 200, 100)
                        if biomass < 30:
                            biomass_color = (200, 120, 80)
                        data_items.append(("  Biomass Index:", f"{biomass:.4f}%", biomass_color))

                    if "atmospheric_pressure" in self._climate_averages:
                        pressure = self._climate_averages["atmospheric_pressure"]
                        data_items.append(("  Atm. Pressure:", f"{pressure:.1f} hPa", (160, 190, 220)))

                if self._metrics:
                    data_items.append(("Population:", "", (180, 200, 220)))

                    if 'num_flora' in self._metrics:
                        data_items.append(("  Flora:", f"{self._metrics['num_flora']}", (120, 200, 120)))

                    if 'num_fauna' in self._metrics:
                        data_items.append(("  Fauna:", f"{self._metrics['num_fauna']}", (200, 150, 100)))

                    if 'total_entities' in self._metrics:
                        data_items.append(
                            ("  Total:", f"{self._metrics['num_flora'] + self._metrics['num_fauna']}", (180, 180, 220)))

                    if 'avg_stress' in self._metrics:
                        stress_value = self._metrics['avg_stress']
                        stress_color = (100, 200, 100)
                        if stress_value > 70:
                            stress_color = (200, 100, 100)
                        elif stress_value > 30:
                            stress_color = (200, 200, 100)
                        data_items.append(("  Avg Stress:", f"{stress_value:.2f}", stress_color))

                    if 'avg_toxicity' in self._metrics:
                        tox_value = self._metrics['avg_toxicity']
                        tox_color = (100, 200, 100)
                        if tox_value > 50:
                            tox_color = (200, 100, 100)
                        elif tox_value > 20:
                            tox_color = (200, 200, 100)
                        data_items.append(("  Avg Toxicity:", f"{tox_value:.2f}", tox_color))

                    if 'avg_size' in self._metrics:
                        data_items.append(("  Avg Size:", f"{self._metrics['avg_size']:.2f}", (150, 190, 220)))

                y_left = self._render_info_section(self._surface, "Biome", data_items, x_left + 5, y_next,
                                                   col_width - 10)
            else:
                y_left = y_next

        if hasattr(self, '_climate_analysis') and self._climate_analysis:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "CLIMATE ANALYSIS", x_left, y_left + 10, "climate_analysis")

            if expanded:
                y_left = self._render_climate_analysis(
                    self._surface, self._climate_analysis, x_left + 5, y_next, col_width - 10)
            else:
                y_left = y_next

        if self._biome_score:
            y_next, expanded = self._render_collapsible_section(
                self._surface, "BIOME SCORE", x_left, y_left + 10, "puntuacion")

            if expanded:
                quality = self._biome_score["quality"]
                quality_color = self._quality_colors.get(quality, self._text_color)

                contributor_count = len(self._biome_score.get('contributor_scores', {}))
                section_height = 120 + (contributor_count * 30)

                section_rect = pygame.Rect(x_left, y_next, col_width, section_height)
                pygame.draw.rect(self._surface, (20, 30, 45), section_rect)
                pygame.draw.rect(self._surface, (40, 70, 110), section_rect, 1)

                score_value = self._biome_score['score']
                score_value = min(10.0, score_value)
                score_percent = score_value / 10.0

                quality_text = quality.upper()
                quality_font = pygame.font.SysFont(None, self._font_size + 12)
                quality_surface = quality_font.render(quality_text, True, quality_color)
                quality_rect = quality_surface.get_rect(centerx=x_left + col_width // 2, y=y_next + 15)
                self._surface.blit(quality_surface, quality_rect)

                score_font = pygame.font.SysFont(None, self._font_size + 20)
                score_surface = score_font.render(f"{score_value:.1f}/10", True, quality_color)
                score_rect = score_surface.get_rect(centerx=x_left + col_width // 2, y=y_next + 50)
                self._surface.blit(score_surface, score_rect)

                bar_width = col_width - 40
                bar_height = 12
                bar_x = x_left + 20
                bar_y = y_next + 85

                pygame.draw.rect(self._surface, (40, 40, 50),
                                 pygame.Rect(bar_x, bar_y, bar_width, bar_height))

                bar_fill_width = int(bar_width * score_percent)
                pygame.draw.rect(self._surface, quality_color,
                                 pygame.Rect(bar_x, bar_y, bar_fill_width, bar_height))
                pygame.draw.rect(self._surface, (80, 100, 120),
                                 pygame.Rect(bar_x, bar_y, bar_width, bar_height), 1)

                current_y = y_next + 110

                factors_title = self._font_bold.render("Contributing Factors:", True, (180, 200, 220))
                self._surface.blit(factors_title, (x_left + 20, current_y))
                current_y += 25

                if 'contributor_scores' in self._biome_score:
                    contributor_scores = self._biome_score['contributor_scores']

                    normalized_scores = {k: min(1.0, v) for k, v in contributor_scores.items()}

                    max_label_width = 0
                    for factor in normalized_scores.keys():
                        factor_name = factor.replace('_', ' ').title() + ":"
                        label_surface = self._font.render(factor_name, True, (180, 200, 220))
                        max_label_width = max(max_label_width, label_surface.get_width())

                    max_label_width = min(max_label_width, col_width // 2 - 10)

                    for factor, score in normalized_scores.items():
                        factor_name = factor.replace('_', ' ').title() + ":"

                        factor_color = (100, 200, 100)
                        if score < 0.3:
                            factor_color = (200, 100, 100)
                        elif score < 0.6:
                            factor_color = (200, 200, 100)

                        label_surface = self._font.render(factor_name, True, (180, 200, 220))

                        if label_surface.get_width() > max_label_width:
                            truncated_name = factor_name
                            while label_surface.get_width() > max_label_width - 10:
                                truncated_name = truncated_name[:-2] + ":"
                                label_surface = self._font.render(truncated_name, True, (180, 200, 220))

                        self._surface.blit(label_surface, (x_left + 20, current_y))

                        value_text = f"{score:.2f}"
                        value_surface = self._font.render(value_text, True, factor_color)
                        value_x = x_left + col_width - 20 - value_surface.get_width()
                        self._surface.blit(value_surface, (value_x, current_y))

                        current_y += 25

                y_left = y_next + section_height
            else:
                y_left = y_next
        y_next, expanded = self._render_collapsible_section(
            self._surface, "ECOLOGICAL METRICS", x_left, y_left + 10, "ecological_metrics")

        if expanded:
            eco_data_items = []

            if 'climate_adaptation' in self._metrics:
                adaptation = self._metrics['climate_adaptation']
                adaptation_color = (100, 200, 100)
                if adaptation < 0.3:
                    adaptation_color = (200, 100, 100)
                elif adaptation < 0.7:
                    adaptation_color = (200, 200, 100)
                eco_data_items.append(("Climate Adaptation:", f"{adaptation:.2f}", adaptation_color))

            if 'entity_balance' in self._metrics:
                balance = self._metrics['entity_balance']
                balance_color = (100, 200, 100)
                if balance < 0.3:
                    balance_color = (200, 100, 100)
                elif balance < 0.7:
                    balance_color = (200, 200, 100)
                eco_data_items.append(("Species Balance:", f"{balance:.2f}", balance_color))

            if any(f'evolution_trend_{i}' in self._metrics for i in range(1, 4)):
                eco_data_items.append(("Evolution Trends:", "", (180, 200, 220)))

                for i in range(1, 4):
                    trend_key = f'evolution_trend_{i}'
                    direction_key = f'evolution_trend_{i}_direction'

                    if trend_key in self._metrics and self._metrics[trend_key]:
                        trend_text = self._metrics[trend_key]
                        direction = self._metrics.get(direction_key, 0)

                        if direction > 0:
                            color = (100, 200, 255)
                        elif direction < 0:
                            color = (255, 120, 120)
                        else:
                            color = (180, 180, 180)

                        label = f"{i}:"
                        eco_data_items.append((label, trend_text, color))

            y_left = self._render_info_section(self._surface, "Ecosystem Metrics",
                                               eco_data_items, x_left + 5, y_next, col_width - 10)
        else:
            y_left = y_next

        if self._selected_entity:
            is_dead = False
            if isinstance(self._selected_entity.state_fields, dict) and "general" in self._selected_entity.state_fields:
                general_fields = self._selected_entity.state_fields.get("general", {})
                if isinstance(general_fields, dict):
                    is_dead = general_fields.get("is_dead", False)

            title = "SELECTED ENTITY"
            if is_dead:
                title = "DEAD ENTITY"

            y_next, expanded = self._render_collapsible_section(
                self._surface, title, x_right, y_right, "entity")

            if expanded:
                header_height = 120
                header_rect = pygame.Rect(x_right, y_next, col_width, header_height)
                header_bg = (25, 35, 50)
                if is_dead:
                    header_bg = (50, 25, 25)

                pygame.draw.rect(self._surface, header_bg, header_rect)
                pygame.draw.rect(self._surface, (60, 90, 120) if not is_dead else (120, 60, 60), header_rect, 1)

                id_font = pygame.font.SysFont(None, self._font_size + 10)
                id_text = f"ID: {self._selected_entity.id}"
                if hasattr(self._selected_entity, 'evolution_cycle') and self._selected_entity.evolution_cycle >= 0:
                    id_text += f" | Generation: {self._selected_entity.evolution_cycle}"

                id_surface = id_font.render(id_text, True, (200, 220, 240) if not is_dead else (240, 180, 180))
                self._surface.blit(id_surface, (x_right + 15, y_next + 12))

                type_color = (120, 200, 120) if self._selected_entity.type == "flora" else (200, 150, 100)
                if is_dead:
                    type_color = (220, 150, 150)

                type_text = f"Type: {str(self._selected_entity.type).capitalize()}"
                if self._selected_entity.type == "fauna" and hasattr(self._selected_entity, 'diet_type'):
                    type_text += f" | Diet: {str(self._selected_entity.diet_type).capitalize()}"

                type_surface = self._font.render(type_text, True, type_color)
                self._surface.blit(type_surface, (x_right + 15, y_next + 42))

                species_text = f"Species: {str(self._selected_entity.species).capitalize()}"
                species_surface = self._font.render(species_text, True, type_color)
                self._surface.blit(species_surface, (x_right + 15, y_next + 67))

                position_text = f"Position: {self._selected_entity.position}"
                position_surface = self._font.render(position_text, True, (180, 200, 220))
                self._surface.blit(position_surface, (x_right + 15, y_next + 92))

                if self._selected_entity.habitats:
                    habitats_str = ", ".join(self._selected_entity.habitats)
                    habitats_text = f"Habitats: {habitats_str}"
                    habitats_surface = self._font.render(habitats_text, True, (180, 200, 220))

                    if y_next + 117 + habitats_surface.get_height() <= y_next + header_height:
                        self._surface.blit(habitats_surface, (x_right + 15, y_next + 117))

                y_right = y_next + header_height + 5

                component_order = [
                    "general",
                    "vital",
                    "growth",
                    "photosynthetic_metabolism",
                    "autotrophic_nutrition",
                    "heterotrophic_nutrition",
                    "weather_adaptation"
                ]

                fields_by_component = self._selected_entity.state_fields
                ordered_components = []

                for component_name in component_order:
                    if component_name in fields_by_component:
                        ordered_components.append((component_name, fields_by_component[component_name]))

                for component_name, fields in ordered_components:
                    if not fields:
                        continue

                    component_title = component_name.replace('_', ' ').title()
                    if component_title == "Weather Adaptation Component":
                        component_title = "Weather Adaptation"

                    section_key = component_name.lower()

                    y_next, expanded = self._render_collapsible_section(
                        self._surface, f"-- {component_title} --", x_right, y_right + 5, section_key)

                    if expanded:
                        y_right = self._render_attribute_section(self._surface, fields, x_right + 5, y_next,
                                                                 col_width - 10)
                    else:
                        y_right = y_next
            else:
                y_right = y_next

        elif self._selected_terrain:
            y_right = self._render_text_line(self._surface,
                                             "SELECTED TERRAIN",
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

    def _render_attribute_section(self, surface, fields, x, y, col_width):
        if not fields:
            return y

        field_height = 26

        sorted_fields = []
        bar_fields = []
        normal_fields = []

        for field, value in fields.items():
            if field.lower() in ["vitality", "energy_reserves"]:
                bar_fields.append((field, value))
            else:
                normal_fields.append((field, value))

        bar_fields.sort(key=lambda item: item[0].lower())
        normal_fields.sort(key=lambda item: item[0].lower())

        sorted_fields = bar_fields + normal_fields

        extra_space = 8 if bar_fields else 0
        section_height = len(sorted_fields) * field_height + 6 + extra_space

        section_rect = pygame.Rect(x - 5, y - 3, col_width + 10, section_height)
        pygame.draw.rect(surface, (15, 25, 40), section_rect)
        pygame.draw.rect(surface, (40, 60, 90), section_rect, 1)

        current_y = y + 3

        for i, (field, value) in enumerate(sorted_fields):
            if value is None:
                continue

            is_transition = i == len(bar_fields) and bar_fields and normal_fields

            row_bg_color = (20, 30, 45) if i % 2 == 0 else (25, 35, 50)
            row_rect = pygame.Rect(x, current_y - 2, col_width, field_height)
            pygame.draw.rect(surface, row_bg_color, row_rect)

            field_name = field.replace('_', ' ').title()
            formatted_value = self._format_field_value(field, value)

            is_important = field.lower() in ["vitality", "energy_reserves", "stress_level", "toxicity"]

            text_color = self._text_color
            if is_important:
                if isinstance(value, (int, float)):
                    max_value = 100.0

                    if field.lower() == "vitality" and "max_vitality" in fields:
                        max_value = fields["max_vitality"]
                    elif field.lower() == "energy_reserves" and "max_energy_reserves" in fields:
                        max_value = fields["max_energy_reserves"]
                    elif field.lower() == "toxicity":
                        max_value = 1.0

                    percentage = value / max_value if max_value > 0 else 0

                    if field.lower() in ["stress_level", "toxicity"]:
                        if percentage < 0.3:
                            text_color = (100, 220, 100)
                        elif percentage > 0.7:
                            text_color = (220, 100, 100)
                        else:
                            text_color = (220, 220, 100)
                    else:
                        if percentage < 0.3:
                            text_color = (220, 100, 100)
                        elif percentage > 0.7:
                            text_color = (100, 220, 100)
                        else:
                            text_color = (220, 220, 100)

            text_y = current_y + (field_height - self._font.get_height()) // 2

            field_text = self._font.render(f" {field_name}:", True,
                                           (170, 200, 220) if is_important else (160, 180, 200))
            surface.blit(field_text, (x + 5, text_y))

            value_text = self._font.render(formatted_value, True, text_color)
            value_x = x + col_width - value_text.get_width() - 10
            surface.blit(value_text, (value_x, text_y))

            if field.lower() in ["vitality", "energy_reserves"] and isinstance(value, (int, float)):
                max_field = None
                max_value = 100.0

                if field.lower() == "vitality" and "max_vitality" in fields:
                    max_field = "max_vitality"
                    max_value = fields["max_vitality"]
                elif field.lower() == "energy_reserves" and "max_energy_reserves" in fields:
                    max_field = "max_energy_reserves"
                    max_value = fields["max_energy_reserves"]

                bar_y = current_y + field_height - 6
                bar_width = col_width - 20
                bar_height = 4

                bar_bg_rect = pygame.Rect(x + 10, bar_y, bar_width, bar_height)
                pygame.draw.rect(surface, (40, 40, 50), bar_bg_rect)

                progress = max(0, min(1, value / max_value if max_value > 0 else 0))
                bar_fg_width = int(bar_width * progress)

                if progress < 0.3:
                    bar_color = (180, 60, 60)
                elif progress < 0.6:
                    bar_color = (180, 180, 60)
                else:
                    bar_color = (60, 180, 60)

                bar_fg_rect = pygame.Rect(x + 10, bar_y, bar_fg_width, bar_height)
                pygame.draw.rect(surface, bar_color, bar_fg_rect)
                pygame.draw.rect(surface, (70, 70, 90), bar_bg_rect, 1)

            if is_transition:
                current_y += field_height + extra_space
            else:
                current_y += field_height

        return current_y + 5

    def _render_collapsible_section(self, surface, title, x, y, section_key):
        expanded = self._section_states.get(section_key, True)
        button_text = "[-]" if expanded else "[+]"

        is_dead_entity = title == "DEAD ENTITY"

        title_bg_color = (30, 60, 90)
        title_highlight_color = (80, 130, 180)
        title_text_color = (140, 190, 230)

        if is_dead_entity:
            title_bg_color = (90, 30, 30)
            title_highlight_color = (180, 80, 80)
            title_text_color = (230, 170, 170)

        title_width = self._width // 2 - 20

        title_height = self._font_title.get_height() + 6

        title_bg_rect = pygame.Rect(x - 5, y - 3, title_width, title_height)
        pygame.draw.rect(surface, title_bg_color, title_bg_rect)
        pygame.draw.rect(surface, (60, 100, 140) if not is_dead_entity else (140, 60, 60), title_bg_rect, 1)

        highlight_rect = pygame.Rect(x - 5, y - 3, title_width, 2)
        pygame.draw.rect(surface, title_highlight_color, highlight_rect)

        title_surface = self._font_title.render(title, True, title_text_color)
        title_y = y + (title_height - title_surface.get_height()) // 2 - 3
        surface.blit(title_surface, (x, title_y))

        button_x = x + title_width - 25
        button_y = title_y
        button_surface = self._font.render(button_text, True, (180, 180, 180))
        button_rect = button_surface.get_rect(topleft=(button_x, button_y))
        surface.blit(button_surface, button_rect)

        self._section_buttons[section_key] = button_rect

        return y + title_height + 8, expanded

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height
