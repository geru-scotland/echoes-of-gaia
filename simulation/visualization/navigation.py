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
Interactive navigation controls for snapshot visualization.

Provides button controls for timeline navigation with play functionality;
includes draggable slider for direct snapshot selection.
Manages playback state and updates button availability - handles
user interactions for temporal biome data exploration.
"""

import logging
from typing import Tuple, Callable, Dict, Optional

import pygame

from simulation.visualization.types import Color, Point, Size


class Button:
    def __init__(
            self,
            position: Point,
            size: Size,
            text: str,
            action: Callable[[], None],
            text_color: Color = (220, 220, 225),
            bg_color: Color = (35, 35, 40),
            button_bg_color: Color = (50, 50, 55),
            button_hover_color: Color = (70, 70, 80),
            hover_color: Color = (80, 80, 80),
            disabled_color: Color = (30, 30, 30),
            font_size: int = 24,
            enabled: bool = True
    ):
        self._position = position
        self._size = size
        self._text = text
        self._action = action
        self._text_color = text_color
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._disabled_color = disabled_color
        self._font_size = font_size
        self._enabled = enabled
        self._hovered = False
        self._rect = pygame.Rect(position, size)

        pygame.font.init()
        self._font = pygame.font.SysFont(None, font_size)

    def update(self, mouse_pos: Point) -> None:
        if self._enabled:
            self._hovered = self._rect.collidepoint(mouse_pos)
        else:
            self._hovered = False

    def render(self, surface: pygame.Surface) -> None:
        if not self._enabled:
            bg_color = self._disabled_color
        elif self._hovered:
            bg_color = (60, 120, 160)
        else:
            bg_color = (30, 60, 90)

        pygame.draw.rect(surface, bg_color, self._rect)
        pygame.draw.rect(surface, (50, 100, 130), self._rect, 1)

        text_surface = self._font.render(self._text, True, (180, 220, 230))
        text_rect = text_surface.get_rect(center=self._rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self._enabled:
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._hovered:
                self._action()
                return True

        return False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value


class Navigation:
    def __init__(
            self,
            position: Point,
            size: Size,
            button_size: Size,
            prev_callback: Callable[[], None],
            next_callback: Callable[[], None],
            play_callback: Callable[[], None],
            slider_callback: Callable[[int], None],
            total_snapshots: int = 1,
            text_color: Color = (255, 255, 255),
            bg_color: Color = (30, 30, 30),
            font_size: int = 16,
            text_position: Optional[Point] = None
    ):
        self._logger = logging.getLogger("navigation")
        self._position = position
        self._size = size
        self._button_size = button_size
        self._prev_callback = prev_callback
        self._next_callback = next_callback
        self._play_callback = play_callback
        self._slider_callback = slider_callback
        self._total_snapshots = max(1, total_snapshots)
        self._current_snapshot = 0
        self._text_color = text_color
        self._bg_color = bg_color
        self._font_size = font_size
        self._is_playing = False
        self._text_position = text_position

        pygame.font.init()
        self._font = pygame.font.SysFont(None, font_size)

        self._surface = pygame.Surface(size)

        self._create_buttons()

        self._slider_rect = pygame.Rect(
            self._button_size[0] * 3 + 40,
            size[1] // 2 - 5,
            size[0] - (self._button_size[0] * 3 + 50),
            10
        )

        self._slider_handle_rect = pygame.Rect(
            self._slider_rect.x,
            self._slider_rect.y - 5,
            20,
            20
        )

        self._slider_dragging = False

    def _create_buttons(self) -> None:
        button_width, button_height = self._button_size

        self._prev_button = Button(
            position=(10, self._size[1] // 2 - button_height // 2),
            size=self._button_size,
            text="<",
            action=self._prev_action,
            text_color=self._text_color,
            font_size=self._font_size + 8,
            enabled=self._current_snapshot > 0
        )

        self._play_button = Button(
            position=(20 + button_width, self._size[1] // 2 - button_height // 2),
            size=self._button_size,
            text="▶" if not self._is_playing else "⏸",
            action=self._play_action,
            text_color=self._text_color,
            font_size=self._font_size + 8
        )

        self._next_button = Button(
            position=(30 + button_width * 2, self._size[1] // 2 - button_height // 2),
            size=self._button_size,
            text=">",
            action=self._next_action,
            text_color=self._text_color,
            font_size=self._font_size + 8,
            enabled=self._current_snapshot < self._total_snapshots - 1
        )

    def _prev_action(self) -> None:
        if self._current_snapshot > 0:
            self._prev_callback()

    def _next_action(self) -> None:
        if self._current_snapshot < self._total_snapshots - 1:
            self._next_callback()

    def _play_action(self) -> None:
        self._is_playing = not self._is_playing
        self._play_button._text = "⏸" if self._is_playing else "▶"
        self._play_callback()

    def _update_slider_position(self) -> None:
        if self._total_snapshots <= 1:
            return

        slider_width = self._slider_rect.width - self._slider_handle_rect.width
        relative_pos = self._current_snapshot / (self._total_snapshots - 1)

        self._slider_handle_rect.x = self._slider_rect.x + int(slider_width * relative_pos)

    def _handle_slider_drag(self, mouse_pos: Point) -> None:
        if not self._slider_dragging:
            return

        slider_min_x = self._slider_rect.x
        slider_max_x = self._slider_rect.right - self._slider_handle_rect.width

        new_x = max(slider_min_x, min(mouse_pos[0] - self._slider_handle_rect.width // 2, slider_max_x))
        self._slider_handle_rect.x = new_x

        slider_width = self._slider_rect.width - self._slider_handle_rect.width
        relative_pos = (new_x - slider_min_x) / slider_width
        snapshot_index = int(round(relative_pos * (self._total_snapshots - 1)))

        if snapshot_index != self._current_snapshot:
            self._current_snapshot = snapshot_index
            self._slider_callback(snapshot_index)

    def set_total_snapshots(self, total: int) -> None:
        self._total_snapshots = max(1, total)
        self._current_snapshot = min(self._current_snapshot, self._total_snapshots - 1)
        self._update_slider_position()

        self._prev_button.enabled = self._current_snapshot > 0
        self._next_button.enabled = self._current_snapshot < self._total_snapshots - 1

    def set_current_snapshot(self, index: int) -> None:
        if 0 <= index < self._total_snapshots:
            self._current_snapshot = index
            self._update_slider_position()

            self._prev_button.enabled = self._current_snapshot > 0
            self._next_button.enabled = self._current_snapshot < self._total_snapshots - 1

    def update(self, mouse_pos: Point) -> None:
        rel_mouse_pos = (mouse_pos[0] - self._position[0], mouse_pos[1] - self._position[1])

        self._prev_button.update(rel_mouse_pos)
        self._play_button.update(rel_mouse_pos)
        self._next_button.update(rel_mouse_pos)

        self._handle_slider_drag(rel_mouse_pos)

    def render(self, surface: pygame.Surface) -> None:
        self._surface.fill((10, 10, 15))

        self._prev_button.render(self._surface)
        self._play_button.render(self._surface)
        self._next_button.render(self._surface)

        pygame.draw.rect(self._surface, (40, 70, 100), self._slider_rect)
        pygame.draw.rect(self._surface, (80, 160, 200), self._slider_handle_rect, border_radius=10)

        text = f"Snapshot: {self._current_snapshot + 1}/{self._total_snapshots}"
        text_surface = self._font.render(text, True, (180, 220, 230))

        text_rect = text_surface.get_rect(
            left=self._slider_rect.right + 80,
            centery=self._size[1] // 2
        )

        right_margin = self._size[0] - 50
        if text_rect.right > right_margin:
            text_rect.right = right_margin

        self._surface.blit(text_surface, text_rect)

        surface.blit(self._surface, self._position)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if hasattr(event, 'pos'):
            rel_pos = (event.pos[0] - self._position[0], event.pos[1] - self._position[1])
            event_copy = pygame.event.Event(event.type, {'pos': rel_pos, 'button': event.button})
        else:
            event_copy = event

        if self._prev_button.handle_event(event_copy):
            return True

        if self._play_button.handle_event(event_copy):
            return True

        if self._next_button.handle_event(event_copy):
            return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            slider_rect_expanded = self._slider_handle_rect.inflate(10, 10)
            if slider_rect_expanded.collidepoint(rel_pos):
                self._slider_dragging = True
                return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._slider_dragging:
                self._slider_dragging = False
                return True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self._prev_action()
                return True
            elif event.key == pygame.K_RIGHT:
                self._next_action()
                return True
            elif event.key == pygame.K_SPACE:
                self._play_action()
                return True

        return False

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @is_playing.setter
    def is_playing(self, value: bool) -> None:
        self._is_playing = value
        self._play_button._text = "⏸" if self._is_playing else "▶"