import json
import os

import pygame
import yaml

from utils.loggers import setup_logger
from utils.paths import CONFIG_DIR


class Settings:
    class DefaultSettings:
        def __init__(self, config):
            self._loggers = {}
            self.screen_width, self.screen_height = config.get_resolution()
            self._setup_logger()

        def _setup_logger(self):
            self._loggers["game"] = setup_logger("game", "logs/game.log")
            self._loggers["bioma"] = setup_logger("bioma", "logs/bioma.log")
            self._loggers["research"] = setup_logger("research", "logs/research.log")

        def get_logger(self, name):
            try:
                return self._loggers[name]
            except KeyError:
                print(f"Logger '{name}' not found in the settings.")

    class GameSettings(DefaultSettings):
        def __init__(self, config):
            super().__init__(config)
            # self.title = config.get("title")

    class SceneSettings(DefaultSettings):
        def __init__(self, config):
            super().__init__(config)
            self.scene_data = {}

        def load_scene_data(self, scene_name):
            file_path = os.path.join("assets/data/scenes", f"{scene_name}.json")
            with open(file_path, "r") as file:
                self.scene_data = json.load(file)

        def get_scene_data(self):
            return self.scene_data

    class Config:
        def __init__(self, config_path="config.yaml"):
            file_path = os.path.join(CONFIG_DIR, config_path)

            with open(file_path, 'r') as stream:
                self.config = yaml.safe_load(stream) or {}
                if not isinstance(self.config, dict):
                    raise ValueError("Configuration file is not a valid YAML dictionary.")

        def get(self, key):
            return self.config[key]

        def update_resolution(self, width, height):
            self.config["screen_width"] = width
            self.config["screen_height"] = height

        def get_resolution(self):
            screen_width = 0
            screen_height = 0
            try:
                screen_width = self.config["screen_width"]
                screen_height = self.config["screen_height"]
            except Exception as e:
                if not self.config.get("screen_width") or not self.config.get("screen_height"):
                    screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
            return screen_width, screen_height

    def __init__(self):
        self._config = self.Config()

    @property
    def game_settings(self):
        return self.GameSettings(self._config)

    @property
    def scene_settings(self):
        return self.SceneSettings(self._config)
