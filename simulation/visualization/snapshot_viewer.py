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
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import pygame

from simulation.visualization.entity_renderer import EntityRenderer, EntityInfo
from simulation.visualization.info_panel import InfoPanel
from simulation.visualization.map_renderer import MapRenderer
from simulation.visualization.navigation import Navigation
from simulation.visualization.snapshot_loader import SnapshotLoader
from simulation.visualization.types import (
    SnapshotData, Point, Size, ViewerConfig, Color, TerrainTypeInfo
)


class SnapshotViewer:

    def __init__(self, config: ViewerConfig):
        self._logger = logging.getLogger("snapshot_viewer")
        self._config = config
        self._running = False
        self._paused = True
        self._last_frame_time = 0

        pygame.init()

        self._window_size = config["window_size"]
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption(config["title"])

        self._init_components()

        self._loader = SnapshotLoader(Path(config["snapshot_path"]))
        if not self._loader.load():
            self._logger.error(f"No se pudieron cargar los snapshots de {config['snapshot_path']}")

        self._navigation.set_total_snapshots(self._loader.get_snapshot_count())

        self._load_snapshot(self._loader.get_current_snapshot())

    def _init_components(self) -> None:
        cell_size = self._config["cell_size"]
        panel_width = self._config["panel_width"]

        window_width, window_height = self._window_size
        map_area_width = window_width - panel_width

        self._map_renderer = MapRenderer(
            cell_size=cell_size,
            grid_color=(50, 50, 50),
            terrain_colors=self._config["terrain_colors"],
            show_grid=True
        )

        self._entity_renderer = EntityRenderer(
            cell_size=cell_size,
            entity_colors=self._config["entity_colors"]
        )

        self._info_panel = InfoPanel(
            width=panel_width,
            height=window_height - 60,
            background_color=(30, 30, 30),
            text_color=(200, 200, 200),
            font_size=self._config["font_size"]
        )

        self._navigation = Navigation(
            position=(0, window_height - 60),
            size=(window_width, 60),
            button_size=(40, 40),
            prev_callback=self._prev_snapshot,
            next_callback=self._next_snapshot,
            play_callback=self._toggle_play,
            slider_callback=self._go_to_snapshot,
            total_snapshots=1,
            text_color=(200, 200, 200),
            bg_color=(20, 20, 20),
            font_size=self._config["font_size"]
        )

        self._selected_cell: Optional[Tuple[int, int]] = None
        self._selected_entity: Optional[int] = None
        self._map_offset = (0, 0)
        self._dragging = False
        self._drag_start = (0, 0)
        self._last_mouse_pos = (0, 0)

    def _load_snapshot(self, snapshot: Optional[SnapshotData]) -> None:
        if snapshot is None:
            self._logger.warning("Intentando cargar un snapshot nulo")
            return

        try:
            if "terrain" not in snapshot:
                self._logger.error("Error al cargar el snapshot: 'terrain' no encontrado")
                return

            terrain_data = snapshot["terrain"]
            if not terrain_data:
                self._logger.error("Error al cargar el snapshot: datos de terreno vacíos")
                return

            self._map_renderer.set_map_data(terrain_data)

            if "entities" in snapshot:
                self._entity_renderer.set_entities_data(snapshot["entities"])

            if "simulation_time" in snapshot:
                self._info_panel.set_simulation_time(snapshot["simulation_time"])

            if "metrics" in snapshot:
                self._info_panel.set_metrics(snapshot["metrics"])

            if "biome_score" in snapshot:
                self._info_panel.set_biome_score(snapshot["biome_score"])

            self._selected_cell = None
            self._selected_entity = None
            self._entity_renderer.select_entity(None)
            self._info_panel.set_selected_entity(None)
            self._info_panel.set_selected_terrain(None)

            self._logger.info(f"Snapshot {snapshot['snapshot_id']} cargado correctamente")
        except Exception as e:
            self._logger.error(f"Error al cargar el snapshot: {e}", exc_info=True)

    def _prev_snapshot(self) -> None:
        snapshot = self._loader.previous_snapshot()
        if snapshot:
            self._load_snapshot(snapshot)
            self._navigation.set_current_snapshot(self._loader.get_current_index())

    def _next_snapshot(self) -> None:
        snapshot = self._loader.next_snapshot()
        if snapshot:
            self._load_snapshot(snapshot)
            self._navigation.set_current_snapshot(self._loader.get_current_index())

    def _go_to_snapshot(self, index: int) -> None:
        snapshot = self._loader.go_to_snapshot(index)
        if snapshot:
            self._load_snapshot(snapshot)

    def _toggle_play(self) -> None:
        self._paused = not self._paused
        self._logger.info(f"Reproducción {'pausada' if self._paused else 'iniciada'}")

    def _handle_click(self, pos: Point) -> None:
        map_area_pos = (pos[0], pos[1])

        cell = self._map_renderer.get_cell_at_pos((map_area_pos[0] - self._map_offset[0],
                                                   map_area_pos[1] - self._map_offset[1]))

        if cell:
            entity_id = self._entity_renderer.get_entity_at_pos(
                (map_area_pos[0] - self._map_offset[0], map_area_pos[1] - self._map_offset[1]),
                cell
            )

            if entity_id is not None:
                self._selected_entity = entity_id
                self._entity_renderer.select_entity(entity_id)
                entity_info = self._entity_renderer.get_entity_info(entity_id)
                self._info_panel.set_selected_entity(entity_info)
                self._info_panel.set_selected_terrain(None)
                self._logger.info(f"Entidad seleccionada: {entity_id} en posición {cell} (y, x)")
            else:
                self._selected_cell = cell
                self._selected_entity = None
                self._entity_renderer.select_entity(None)
                terrain_info = self._map_renderer.get_terrain_info(cell)
                self._info_panel.set_selected_entity(None)
                self._info_panel.set_selected_terrain(terrain_info)
                self._logger.info(f"Celda seleccionada: {cell} (y, x)")

    def _update(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self._last_mouse_pos = mouse_pos

        if self._dragging:
            dx = mouse_pos[0] - self._drag_start[0]
            dy = mouse_pos[1] - self._drag_start[1]
            self._map_offset = (self._map_offset[0] + dx, self._map_offset[1] + dy)
            self._drag_start = mouse_pos

        self._navigation.update(mouse_pos)
        mouse_pos = pygame.mouse.get_pos()
        panel_width = self._config["panel_width"]
        window_width, window_height = self._window_size

        if (mouse_pos[0] < window_width - panel_width and
                mouse_pos[1] < window_height - 60):
            map_pos = (mouse_pos[0] - self._map_offset[0],
                       mouse_pos[1] - self._map_offset[1])
            self._hover_cell = self._map_renderer.get_cell_at_pos(map_pos)
        else:
            self._hover_cell = None
        if not self._paused and self._navigation.is_playing:
            current_time = time.time()
            if current_time - self._last_frame_time > 1.0:
                self._next_snapshot()
                self._last_frame_time = current_time

    def _render(self) -> None:
        self._screen.fill(self._config["background_color"])

        self._map_renderer.render(self._screen, self._map_offset)
        self._entity_renderer.render(self._screen, self._map_offset)

        if self._selected_cell:
            y, x = self._selected_cell
            cell_size = self._map_renderer.get_cell_size()
            rect = pygame.Rect(
                x * cell_size + self._map_offset[0],
                y * cell_size + self._map_offset[1],
                cell_size,
                cell_size
            )
            pygame.draw.rect(self._screen, (255, 255, 255), rect, 2)

        if self._hover_cell and self._hover_cell != self._selected_cell:
            y, x = self._hover_cell
            cell_size = self._map_renderer.get_cell_size()
            hover_rect = pygame.Rect(
                x * cell_size + self._map_offset[0],
                y * cell_size + self._map_offset[1],
                cell_size,
                cell_size
            )
            pygame.draw.rect(self._screen, (180, 180, 200, 120), hover_rect, 1)

        panel_width = self._config["panel_width"]
        window_width, _ = self._window_size
        self._info_panel.render(self._screen, (window_width - panel_width, 0))
        self._navigation.render(self._screen)

        if self._hover_cell:
            y, x = self._hover_cell
            terrain_info = self._map_renderer.get_terrain_info(self._hover_cell)

            if terrain_info:
                tooltip_text = f"{terrain_info['name']}"
                entity_id = self._entity_renderer.get_entity_at_pos(
                    (self._last_mouse_pos[0] - self._map_offset[0],
                     self._last_mouse_pos[1] - self._map_offset[1]),
                    self._hover_cell
                )

                if entity_id is not None:
                    entity_info = self._entity_renderer.get_entity_info(entity_id)
                    if entity_info:
                        tooltip_text += f" | {entity_info.specific_type}"

                font = pygame.font.SysFont(None, self._config["font_size"])
                text_surface = font.render(tooltip_text, True, (220, 220, 225))
                text_rect = text_surface.get_rect()
                text_rect.bottomleft = (
                    self._last_mouse_pos[0] + 15,
                    self._last_mouse_pos[1] - 5
                )

                bg_rect = text_rect.inflate(10, 6)
                bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
                bg_surface.fill((40, 40, 45, 220))
                self._screen.blit(bg_surface, bg_rect)

                pygame.draw.rect(self._screen, (60, 60, 65), bg_rect, 1)

                self._screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self._prev_snapshot()
                elif event.y < 0:
                    self._next_snapshot()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                panel_width = self._config["panel_width"]
                window_width, window_height = self._window_size

                if event.pos[1] >= window_height - 60:
                    self._navigation.handle_event(event)

                elif event.pos[0] >= window_width - panel_width:
                    pass

                else:
                    if event.button == 1:
                        self._handle_click(event.pos)

                    elif event.button == 2:
                        self._dragging = True
                        self._drag_start = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self._dragging = False

                self._navigation.handle_event(event)

            elif event.type == pygame.MOUSEMOTION:
                if self._dragging:
                    dx = event.pos[0] - self._drag_start[0]
                    dy = event.pos[1] - self._drag_start[1]
                    self._map_offset = (self._map_offset[0] + dx, self._map_offset[1] + dy)
                    self._drag_start = event.pos

    def run(self) -> None:
        self._running = True
        self._paused = True
        clock = pygame.time.Clock()

        self._logger.info("Iniciando el visor de snapshots")

        while self._running:
            self._handle_events()

            self._update()

            self._render()

            clock.tick(self._config["fps"])

        pygame.quit()
        self._logger.info("Visor de snapshots finalizado")