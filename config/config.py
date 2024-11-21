import pygame
import yaml


class Config:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as stream:
            self.config = yaml.safe_load(stream)

    def get(self, key):
        return self.config[key]

    def get_resolution(self):
        screen_width = 0
        screen_height = 0
        try:
            screen_width = self.config["screen_width"]
            screen_height = self.config["screen_height"]
        except KeyError:
            if not self.config.get("screen_width") or not self.config.get("screen_height"):
                screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
        return screen_width, screen_height
