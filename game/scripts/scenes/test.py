
import random

from game.shared.shared import generate_matte_color
from game.systems.scenes.base_scene import Scene


class EntityScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.entities = self.create_entities()
        self.alpha = 0
        self.fade_in = True

    def create_entities(self):
        entities = []
        for _ in range(200):
            size = random.randint(10, 30)
            color = generate_matte_color()
            shape = random.choice(["circle", "square"])
            pos = [random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)]
            speed = [random.uniform(-4, 4), random.uniform(-4, 4)]
            entities.append({"size": size, "color": color, "shape": shape, "pos": pos, "speed": speed})
        return entities

    def handle_events(self, event):
        pass

    def update(self, diff):
        for entity in self.entities:
            entity["pos"][0] += entity["speed"][0]
            entity["pos"][1] += entity["speed"][1]

            if entity["pos"][0] <= 0 or entity["pos"][0] >= SCREEN_WIDTH - entity["size"]:
                entity["speed"][0] *= -1
            if entity["pos"][1] <= 0 or entity["pos"][1] >= SCREEN_HEIGHT - entity["size"]:
                entity["speed"][1] *= -1

    def render(self, screen):
        screen.fill(MATT_BLACK)
        for entity in self.entities:
            if entity["shape"] == "circle":
                pygame.draw.circle(screen, entity["color"], (int(entity["pos"][0]), int(entity["pos"][1])),
                                   entity["size"] // 2)
            elif entity["shape"] == "square":
                pygame.draw.rect(screen, entity["color"], (*entity["pos"], entity["size"], entity["size"]))