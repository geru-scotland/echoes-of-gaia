

class SceneManager:
    def __init__(self):
        self.current_scene = None

    def set_scene(self, scene, args=None, **kwargs):
        self.current_scene = scene(self.on_finished, args, **kwargs)

    def on_finished(self, scene=None, args=None, **kwargs):
        if scene:
            self.set_scene(scene, args, **kwargs)
        print("Scene finished")

    def handle_events(self, event):
        self.current_scene.handle_events(event)

    def update(self, diff):
        self.current_scene.update(diff)

    def render(self, screen):
        self.current_scene.render(screen)