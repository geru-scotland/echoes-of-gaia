"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
import json
import os
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from utils.paths import ASSETS_DIR, CONFIG_DIR


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

    @property
    def config(self):
        return self._config


class GameDisplaySettings:
    def __init__(self, config):
        self._settings = config
        self.screen_width, self.screen_height = self._get_resolution()

    def _get_resolution(self):
        width = self._settings.get("screen_width")
        height = self._settings.get("screen_height")
        if not width or not height:
            import pygame
            width, height = pygame.display.get_desktop_sizes()[0]
        return width, height


class RenderDisplaySettings:
    def __init__(self, config):
        self._settings = config
        self.window_width, self.window_height = self._get_resolution()

    def _get_resolution(self):
        width = self._settings.get("window_width")
        height = self._settings.get("window_height")
        return width, height


class RenderSettings(DefaultSettings, RenderDisplaySettings):
    def __init__(self, config_file="render.yaml"):
        DefaultSettings.__init__(self, config_file)
        RenderDisplaySettings.__init__(self, self.config)
        self.title = self.config.get("title")


class GameSettings(DefaultSettings, GameDisplaySettings):
    def __init__(self, config_file="game.yaml"):
        DefaultSettings.__init__(self, config_file)
        GameDisplaySettings.__init__(self, self.config)
        self.title = self.config.get("title")


class SceneSettings(DefaultSettings, GameDisplaySettings):
    def __init__(self, config_file="game.yaml"):
        DefaultSettings.__init__(self, config_file)
        GameDisplaySettings.__init__(self, self.config)
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

    def get_logger(self, name="biome"):
        return self._loggers.get(name)


class SimulationSettings(DefaultSettings):
    def __init__(self, config_file="simulation.yaml"):
        super().__init__(config_file)
        self._influxdb_config: Dict[str, Any] = self._load_influxdb_config()

    def _load_influxdb_config(self):
        try:
            return {
                "url": os.getenv("INFLUXDB_URL"),
                "token": os.getenv("INFLUXDB_TOKEN"),
                "org": os.getenv("INFLUXDB_ORG"),
                "bucket": os.getenv("INFLUXDB_BUCKET")
            }
        except:
            raise

    @property
    def influxdb_config(self) -> Dict[str, Any]:
        return self._influxdb_config

    def get_logger(self, name="simulation"):
        return self._loggers.get(name)


class Settings:
    def __init__(self, override_configs=None):
        load_dotenv()
        self._log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        self._override_configs = override_configs or {}

        self._game_settings = None
        self._scene_settings = None
        self._biome_settings = None
        self._simulation_settings = None
        self._render_settings = None

    @property
    def log_level(self):
        return self._log_level

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
            config_file = self._override_configs if self._override_configs else "biome.yaml"
            self._biome_settings = BiomeSettings(config_file)
        return self._biome_settings

    @property
    def simulation_settings(self):
        if self._simulation_settings is None:
            config_file = self._override_configs if self._override_configs else "simulation.yaml"
            self._simulation_settings = SimulationSettings(config_file)
        return self._simulation_settings

    @property
    def render_settings(self):
        if self._render_settings is None:
            self._render_settings = RenderSettings()
        return self._render_settings
