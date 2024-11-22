from utils.dependecy_injector import dependency_injector


class Scene:
    def __init__(self, scene_name, args=None, **kwargs):
        try:
            self.settings = dependency_injector.get("scene_settings")
            self.settings.load_scene_data(scene_name)
            self._scene_data = self.settings.get_scene_data()
            self._screen_width, self._screen_height = self.settings.screen_width, self.settings.screen_height
            self._logger = self.settings.get_logger("game")
        except Exception as e:
            print(f"Critical error: Missing dependency - {e}")

    def handle_events(self, event):
        pass

    def update(self, diff):
        pass

    def render(self, screen):
        pass
