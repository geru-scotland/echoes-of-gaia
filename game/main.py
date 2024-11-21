from enum import Enum

import pygame
import sys

from config.config import Config
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
        self.config = dependency_injector.get("config")
        pygame.init()
        pygame.mixer.init()
        screen_width, screen_height = self.config.get_resolution()
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene_manager = SceneManager(scene=IntroScene)

        pygame.mixer.music.load(AUDIO_FILE)
        pygame.mixer.music.play(-1)

    def run(self):
        while self.running:
            diff = self.clock.tick(60)
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
    dependency_injector.register("config", Config())
    game = Game()
    game.run()
