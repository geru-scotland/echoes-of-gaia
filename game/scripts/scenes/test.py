from enum import Enum
import pygame
import random
from game.theme.colors import Colors
from game.systems.scenes.base_scene import Scene


class EntitySceneState(Enum):
    STATE_FADE_IN = 0
    STATE_ACTIVE = 1


class EntityScene(Scene):
    def __init__(self, on_finish_callback=None, args=None, **kwargs):
        super().__init__(__class__.__name__)
        self.entities = self.create_entities()
        self.alpha = 0
        self.state = EntitySceneState.STATE_FADE_IN
        self.fade_in_duration = 3000
        self.fade_timer = self.fade_in_duration

    def create_entities(self):
        entities = []
        for _ in range(200):
            size = random.randint(10, 30)
            color = Colors.generate_matte_color()
            shape = random.choice(["circle", "square"])
            pos = [random.randint(0, self._screen_width), random.randint(0, self._screen_height)]
            speed = [random.uniform(-4, 4), random.uniform(-4, 4)]
            entities.append({"size": size, "color": color, "shape": shape, "pos": pos, "speed": speed})
        return entities

    def handle_events(self, event):
        pass

    def update(self, diff):
        if self.state == EntitySceneState.STATE_FADE_IN:
            self.fade_timer -= diff
            self.alpha = 255 * (1 - max(self.fade_timer / self.fade_in_duration, 0))
            if self.fade_timer <= 0:
                self.state = EntitySceneState.STATE_ACTIVE
                self.alpha = 255

        elif self.state == EntitySceneState.STATE_ACTIVE:
            for entity in self.entities:
                entity["pos"][0] += entity["speed"][0]
                entity["pos"][1] += entity["speed"][1]

                if entity["pos"][0] <= 0 or entity["pos"][0] >= self._screen_width - entity["size"]:
                    entity["speed"][0] *= -1
                if entity["pos"][1] <= 0 or entity["pos"][1] >= self._screen_height - entity["size"]:
                    entity["speed"][1] *= -1

    def render(self, screen):
        screen.fill(Colors.Background.MATT_BLACK)
        for entity in self.entities:
            if entity["shape"] == "circle":
                pygame.draw.circle(screen, entity["color"], (int(entity["pos"][0]), int(entity["pos"][1])),
                                   entity["size"] // 2)
            elif entity["shape"] == "square":
                pygame.draw.rect(screen, entity["color"], (*entity["pos"], entity["size"], entity["size"]))

        if self.state == EntitySceneState.STATE_FADE_IN:
            overlay = pygame.Surface((self._screen_width, self._screen_height))
            overlay.set_alpha(255 - int(self.alpha))
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
