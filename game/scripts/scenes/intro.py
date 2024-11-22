import os
from enum import Enum

import pygame

from game.scripts.scenes.test import EntityScene
from game.systems.scenes.base_scene import Scene
from game.theme.colors import Colors
from utils.paths import ASSETS_DIR


class IntValueEnum(Enum):
    def __int__(self):
        return self.value


class IntroSceneState(Enum):
    STATE_LOADING = 0
    STATE_FADE_IN = 1
    STATE_IDLE = 2
    STATE_FADE_OUT = 3,
    STATE_FINISHED = 4


class IntroSceneTimers(IntValueEnum):
    START_FADE_IN = 2000


# Para transiciones, igual crear una clase especial y que introscene herede de ella
class IntroScene(Scene):
    def __init__(self, on_finish_callback=None, args=None, **kwargs):
        super().__init__(__class__.__name__)

        self._on_finish_callback = on_finish_callback
        self._title = self._scene_data.get("title")
        self._fade_speed = self._scene_data.get("fade-speed")
        print(self._assets)
        self.title_surface = self._assets.get("title-font").render(self._title, True, Colors.Text.PRIMARY_TEXT)
        self.press_key_surface = self._assets.get("title-font").render("Press any key to continue", True, Colors.Text.PRIMARY_TEXT)

        self.text_rect = self.title_surface.get_rect(center=(self._screen_width // 2, self._screen_height // 2))
        self.press_key_rect = self.press_key_surface.get_rect(
            center=(self._screen_width // 2, self._screen_height // 2 + 300))

        self._start_music()

        self.state = IntroSceneState.STATE_LOADING
        self.start_timer = int(IntroSceneTimers.START_FADE_IN)

        self.alpha = 0
        self.blink_alpha = 0
        self.blink_increasing = False
        self.sound_played = False

    def _load_assets(self):
        try:
            assets = {}
            fonts_dir = os.path.join(ASSETS_DIR, "fonts")
            audio_dir = os.path.join(ASSETS_DIR, "audio")
            assets['title-font'] = pygame.font.Font(f"{fonts_dir}/{self._scene_data["title-font"]}", self._scene_data["title-size"])
            assets['small-font'] = pygame.font.Font(None, self._scene_data["small-text-size"])
            assets['main-song'] = pygame.mixer.Sound(f"{audio_dir}/songs/{self._scene_data["audio"]["main-song"]}")
            assets['sound-effect'] = pygame.font.Font(f"{audio_dir}/effects/{self._scene_data["audio"]["sound-effect"]}")
            return assets
        except Exception as e:
            print(f"Error loading assets: {e}")
            return {}

    def _show_texts(self):
        pass

    def _start_music(self):
        pygame.mixer.music.load(self._assets.get("main-song"))
        pygame.mixer.music.play(-1)

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN and self.state == IntroSceneState.STATE_IDLE:
            self.state = IntroSceneState.STATE_FADE_OUT

            if not self.sound_played:
                self._assets.get("sound-effect").play()
                self.sound_played = True

    def update(self, diff):
        # timer con delay del inicio del fade in
        if (self.start_timer <= diff
                and self.state == IntroSceneState.STATE_LOADING):
            self.start_timer = int(IntroSceneTimers.START_FADE_IN)
            self.state = IntroSceneState.STATE_FADE_IN
        else:
            self.start_timer -= diff

        if self.state == IntroSceneState.STATE_FADE_IN:
            self.alpha += self._fade_speed
            if self.alpha >= 255:
                self.alpha = 255
                self.state = IntroSceneState.STATE_IDLE
                self.blink_increasing = True

        elif self.state == IntroSceneState.STATE_IDLE:
            if self.blink_increasing:
                self.blink_alpha += 1
                if self.blink_alpha >= 255:
                    self.blink_alpha = 255
                    self.blink_increasing = False
            else:
                self.blink_alpha -= 2
                if self.blink_alpha <= 2:
                    self.blink_alpha = 50
                    self.blink_increasing = True

        elif self.state == IntroSceneState.STATE_FADE_OUT:
            self.alpha -= self._fade_speed
            if self.alpha <= 0:
                self.alpha = 0
                if self._on_finish_callback:
                    self._on_finish_callback(EntityScene)
                    self.state = IntroSceneState.STATE_FINISHED

    def render(self, screen):
        screen.fill(Colors.Background.DARK_BLACK)

        # renderizo titulo
        self.title_surface.set_alpha(self.alpha)
        screen.blit(self.title_surface, self.text_rect)

        # press key
        if self.state == IntroSceneState.STATE_IDLE:
            self.press_key_surface.set_alpha(self.blink_alpha)
            screen.blit(self.press_key_surface, self.press_key_rect)
        elif self.state == IntroSceneState.STATE_FADE_OUT:
            self.press_key_surface.set_alpha(self.alpha)
            screen.blit(self.press_key_surface, self.press_key_rect)
