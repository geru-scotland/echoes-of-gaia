from enum import Enum

import pygame
import sys

from config.settings import Settings
from game.scripts.scenes.intro import IntroScene
from game.systems.scenes.scene_manager import SceneManager
from utils.dependecy_injector import dependency_injector

SCREEN_WIDTH = None
SCREEN_HEIGHT = None
TITLE = "Echoes of Gaia"
FADE_SPEED = 0.8
AUDIO_FILE = "assets/audio/intro.mp3"

DARK_BLACK = (0, 0, 0)
MATT_BLACK = (30, 30, 30)
BRIGHT_WHITE = (255, 255, 255)


class Game:
    def __init__(self):
        try:
            self.settings = dependency_injector.get("settings")
        except KeyError as e:
            print(f"Critical error: Missing dependency - {e}")
            sys.exit(1)

        self._logger = self.settings.get_logger("game")

        self._logger.info("Initializing game...")

        pygame.init()
        pygame.mixer.init()
        screen_width, screen_height = self.settings.get_resolution()
        self._logger.info(f"Screen resolution: {screen_width}x{screen_height}")
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene_manager = SceneManager(scene=IntroScene)

        pygame.mixer.music.load(AUDIO_FILE)
        pygame.mixer.music.play(-1)

        self._logger.info("Game initialized.")

    def run(self):
        while self.running:
            diff = self.clock.tick(60)  # TODO: Calcular delta time para movimientos, si asumo en px/s, diff/1000
            self.handle_events()
            self.update(diff)
            self.render()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            self.scene_manager.handle_events(event)

    def update(self, diff):
        self.scene_manager.update(diff)

    def render(self):
        self.scene_manager.render(self.screen)
        pygame.display.flip()


if __name__ == "__main__":
    dependency_injector.register("settings", Settings())
    game = Game()
    game.run()
