from game.scripts.scenes.transition import TransitionScene


class SceneManager:
    def __init__(self, scene=None):
        self.current_scene = scene(self)

    def change_scene(self, new_scene, transition_scene=TransitionScene):
        self.current_scene = transition_scene(self, self.current_scene, new_scene)

    def handle_events(self, event):
        self.current_scene.handle_events(event)

    def update(self, diff):
        self.current_scene.update(diff)

    def render(self, screen):
        self.current_scene.render(screen)