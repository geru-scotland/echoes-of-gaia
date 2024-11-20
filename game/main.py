import pygame
import random
import sys


SCREEN_WIDTH = None
SCREEN_HEIGHT = None
TITLE = "Echoes of Gaia"
FADE_SPEED = 1
AUDIO_FILE = "assets/audio/intro.mp3"


DARK_BLACK = (0, 0, 0)
MATT_BLACK = (30, 30, 30)
BRIGHT_WHITE = (255, 255, 255)


def generate_matte_color():
    return random.choice([
        (102, 102, 153),  # Bluish gray
        (102, 153, 102),  # Matte green
        (153, 102, 102),  # Matte red
        (153, 153, 102),  # Matte yellow
        (102, 102, 102)   # Dark matte gray
    ])


class ClickableEntity:
    def __init__(self, x, y, width, height, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.surface = pygame.Surface((width, height))
        self.surface.fill(color)

    def draw(self, screen):
        screen.blit(self.surface, self.rect.topleft)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        global SCREEN_WIDTH, SCREEN_HEIGHT
        SCREEN_WIDTH, SCREEN_HEIGHT = pygame.display.get_desktop_sizes()[0]
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.scene_manager = SceneManager()

        pygame.mixer.music.load(AUDIO_FILE)
        pygame.mixer.music.play(-1)

    def run(self):
        while self.running:
            self.clock.tick(60)
            self.handle_events()
            self.update()
            self.render()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            self.scene_manager.handle_events(event)

    def update(self):
        self.scene_manager.update()

    def render(self):
        self.scene_manager.render(self.screen)
        pygame.display.flip()


class SceneManager:
    def __init__(self):
        self.current_scene = IntroScene(self)

    def change_scene(self, new_scene):
        self.current_scene = TransitionScene(self, self.current_scene, new_scene)

    def handle_events(self, event):
        self.current_scene.handle_events(event)

    def update(self):
        self.current_scene.update()

    def render(self, screen):
        self.current_scene.render(screen)


class Scene:
    def __init__(self, manager):
        self.manager = manager

    def handle_events(self, event):
        pass

    def update(self):
        pass

    def render(self, screen):
        pass


class IntroScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.Font(None, 100)
        self.small_font = pygame.font.Font(None, 30)
        font = pygame.font.Font("assets/fonts/orbitron/Orbitron-VariableFont_wght.ttf", 100)
        self.title_surface = font.render(TITLE, True, BRIGHT_WHITE)
        self.press_key_surface = self.small_font.render("Press any key to continue", True, BRIGHT_WHITE)
        self.alpha = 0
        self.fade_in = True
        self.fade_out = False
        self.blink_alpha = 0
        self.blink_increasing = True
        self.text_rect = self.title_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.press_key_rect = self.press_key_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 300))
        self.sound = pygame.mixer.Sound("assets/audio/effects/ff_menu.ogg")
        self.sound_played = False

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN and not self.fade_out:
            self.fade_in = False
            self.fade_out = True

            if not self.sound_played:
                self.sound.play()
                self.sound_played = True

    def update(self):
        if self.fade_in and self.alpha < 255:
            self.alpha += FADE_SPEED
        elif self.fade_out and self.alpha > 0:
            self.alpha -= FADE_SPEED
            if self.alpha <= 0:
                self.manager.change_scene(EntityScene(self.manager))

        if self.blink_increasing:
            self.blink_alpha += 2
            if self.blink_alpha >= 255:
                self.blink_alpha = 255
                self.blink_increasing = False
        else:
            self.blink_alpha -= 2
            if self.blink_alpha <= 50:
                self.blink_alpha = 50
                self.blink_increasing = True

    def render(self, screen):
        screen.fill(DARK_BLACK)
        self.title_surface.set_alpha(self.alpha)
        self.press_key_surface.set_alpha(self.blink_alpha)
        screen.blit(self.title_surface, self.text_rect)
        screen.blit(self.press_key_surface, self.press_key_rect)


class EntityScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.entities = self.create_entities()
        self.alpha = 0
        self.fade_in = True

    def create_entities(self):
        entities = []
        for _ in range(60):
            size = random.randint(10, 30)
            color = generate_matte_color()
            shape = random.choice(["circle", "square"])
            pos = [random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)]
            speed = [random.uniform(-4, 4), random.uniform(-4, 4)]
            entities.append({"size": size, "color": color, "shape": shape, "pos": pos, "speed": speed})
        return entities

    def handle_events(self, event):
        pass

    def update(self):
        if self.fade_in and self.alpha < 255:
            self.alpha += FADE_SPEED
        else:
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
                pygame.draw.circle(screen, entity["color"], (int(entity["pos"][0]), int(entity["pos"][1])), entity["size"] // 2)
            elif entity["shape"] == "square":
                pygame.draw.rect(screen, entity["color"], (*entity["pos"], entity["size"], entity["size"]))

        if self.fade_in:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(255 - self.alpha)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))


class TransitionScene(Scene):
    def __init__(self, manager, current_scene, next_scene):
        super().__init__(manager)
        self.current_scene = current_scene
        self.next_scene = next_scene
        self.alpha = 0
        self.transitioning_in = False

    def update(self):
        if not self.transitioning_in:
            self.alpha += FADE_SPEED
            if self.alpha >= 255:
                self.transitioning_in = True
                self.current_scene = self.next_scene
        else:
            self.alpha -= FADE_SPEED
            if self.alpha <= 0:
                self.manager.current_scene = self.current_scene

    def render(self, screen):
        if not self.transitioning_in:
            self.current_scene.render(screen)
        else:
            self.next_scene.render(screen)
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(self.alpha)
        overlay.fill((0, 0, 0))
        screen.blit(overlay, (0, 0))


if __name__ == "__main__":
    game = Game()
    game.run()
