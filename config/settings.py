import json
import os
import yaml

from utils.loggers import setup_logger
from utils.paths import CONFIG_DIR, ASSETS_DIR


class Config:
    def __init__(self, config_file):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        file_path = os.path.join(CONFIG_DIR, config_file)
        try:
            with open(file_path, 'r') as stream:
                config_data = yaml.safe_load(stream) or {}
                if not isinstance(config_data, dict):
                    raise ValueError("Configuration file is not a valid YAML dictionary.")
                return config_data
        except FileNotFoundError:
            print(f"Config file not found: {file_path}")
            return {}

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return f"Config({self.config})"


class DefaultSettings:
    def __init__(self, config_file):
        self._config = Config(config_file)
        self._loggers = {}
        self._setup_default_loggers()

    def _setup_default_loggers(self):
        pass

    def get_logger(self, name):
        return self._loggers.get(name, None)

    @property
    def config(self):
        return self._config

class DisplaySettings:
    def __init__(self, config):
        self.config = config
        self.screen_width, self.screen_height = self._get_resolution()

    def _get_resolution(self):
        width = self.config.get("screen_width")
        height = self.config.get("screen_height")
        if not width or not height:
            import pygame
            width, height = pygame.display.get_desktop_sizes()[0]
        return width, height

    def update_resolution(self, width, height):
        self.config.update_resolution(width, height)
        self.screen_width, self.screen_height = width, height

class GameSettings(DefaultSettings, DisplaySettings):
    def __init__(self, config_file="game.yaml"):
        DefaultSettings.__init__(self, config_file)
        DisplaySettings.__init__(self, self.config)
        self._loggers["game"] = setup_logger("game", "game.log")
        self.title = self.config.get("title")


class SceneSettings(DefaultSettings, DisplaySettings):
    def __init__(self, config_file="game.yaml"):
        DefaultSettings.__init__(self, config_file)
        DisplaySettings.__init__(self, self.config)
        self._loggers["scene"] = setup_logger("scene", "scene.log")
        self.scene_data = {}

    def load_scene_data(self, scene_name):
        file_path = os.path.join(f"{ASSETS_DIR}/data/scenes/", f"{scene_name}.json")
        try:
            with open(file_path, "r") as file:
                self.scene_data = json.load(file) or {}
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading scene file: {file_path}")
            self.scene_data = {}
        return self.scene_data


class BiomeSettings(DefaultSettings):
    def __init__(self, config_file="biome.yaml"):
        super().__init__(config_file)
        self._loggers["biome"] = setup_logger("biome", "biome.log")

    def get_logger(self, name="biome"):
        return self._loggers.get(name)


class SimulationSettings(DefaultSettings):
    def __init__(self, config_file="simulation.yaml"):
        super().__init__(config_file)
        self._loggers["simulation"] = setup_logger("simulation", "simulation.log")


class Settings:
    def __init__(self):
        self._game_settings = None
        self._scene_settings = None
        self._biome_settings = None
        self._simulation_settings = None

    @property
    def game_settings(self):
        if self._game_settings is None:
            self._game_settings = GameSettings()
        return self._game_settings

    @property
    def scene_settings(self):
        if self._scene_settings is None:
            self._scene_settings = SceneSettings()
        return self._scene_settings

    @property
    def biome_settings(self):
        if self._biome_settings is None:
            self._biome_settings = BiomeSettings()
        return self._biome_settings

    @property
    def simulation_hsettings(self):
        if self._simulation_settings is None:
            self._simulation_settings = SimulationSettings()
        return self._simulation_settings
