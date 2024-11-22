import os.path
from abc import abstractmethod

from utils.dependecy_injector import dependency_injector
from utils.paths import ASSETS_DIR


class Scene:
    def __init__(self, scene_name, args=None, **kwargs):
        try:
            self.settings = dependency_injector.get("scene_settings")
            self._scene_data = self.settings.load_scene_data(scene_name)
            self._assets = self._load_assets()
            self._screen_width, self._screen_height = self.settings.screen_width, self.settings.screen_height
            self._logger = self.settings.get_logger("game")
        except Exception as e:
            print(f"Critical error: Missing dependency - {e}")

    @abstractmethod
    def _load_assets(self):
        pass

    @abstractmethod
    def handle_events(self, event):
        pass

    @abstractmethod
    def update(self, diff):
        pass

    @abstractmethod
    def render(self, screen):
        pass
