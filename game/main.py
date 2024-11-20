from enum import Enum

import pygame
import random
import sys

SCREEN_WIDTH = None
SCREEN_HEIGHT = None
TITLE = "Echoes of Gaia"
FADE_SPEED = 0.8
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
        (102, 102, 102)  # Dark matte gray
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
        self.scene_manager = SceneManager(scene=IntroScene)

        pygame.mixer.music.load(AUDIO_FILE)
        pygame.mixer.music.play(-1)

    def run(self):
        while self.running:
            diff = self.clock.tick(60)
            self.handle_events()
            self.update(diff)
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

    def update(self, diff):
        self.scene_manager.update(diff)

    def render(self):
        self.scene_manager.render(self.screen)
        pygame.display.flip()


class Scene:
    def __init__(self, manager):
        self.manager = manager

    def handle_events(self, event):
        pass

    def update(self, diff):
        pass

    def render(self, screen):
        pass


# Scene manager haré instancia en game
class TransitionScene(Scene):
    def __init__(self, manager, current_scene, next_scene):
        super().__init__(manager)
        self.current_scene = current_scene
        self.next_scene = next_scene
        self.alpha = 0
        self.transitioning_in = False

    def update(self, diff):
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


class IntValueEnum(Enum):
    def __int__(self):
        return self.value


class IntroSceneState(Enum):
    STATE_LOADING = 0
    STATE_FADE_IN = 1
    STATE_IDLE = 2
    STATE_FADE_OUT = 3


class IntroSceneTimers(IntValueEnum):
    START_FADE_IN = 2000


# Para transiciones, igual crear una clase especial y que introscene herede de ella
class IntroScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.Font(None, 100)
        self.small_font = pygame.font.Font(None, 30)
        font = pygame.font.Font("assets/fonts/orbitron/Orbitron-VariableFont_wght.ttf", 100)
        self.title_surface = font.render(TITLE, True, BRIGHT_WHITE)
        self.press_key_surface = self.small_font.render("Press any key to continue", True, BRIGHT_WHITE)
        self.alpha = 0
        self.blink_alpha = 0
        self.blink_increasing = False
        self.text_rect = self.title_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.press_key_rect = self.press_key_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 300))
        self.sound = pygame.mixer.Sound("assets/audio/effects/ff_menu.ogg")
        self.sound_played = False

        self.state = IntroSceneState.STATE_LOADING
        self.start_timer = int(IntroSceneTimers.START_FADE_IN)

    def handle_events(self, event):
        if event.type == pygame.KEYDOWN and self.state == IntroSceneState.STATE_IDLE:
            self.state = IntroSceneState.STATE_FADE_OUT

            if not self.sound_played:
                self.sound.play()
                self.sound_played = True

    def update(self, diff):

        # timer con delay del inicio del fade in
        if (self.start_timer <= diff
                and self.state == IntroSceneState.STATE_LOADING):
            self.start_timer = int(IntroSceneTimers.START_FADE_IN)
            self.state = IntroSceneState.STATE_FADE_IN
        else:
            self.start_timer -= diff

        if self.state == IntroSceneState.STATE_FADE_IN:
            self.alpha += FADE_SPEED
            if self.alpha >= 255:
                self.alpha = 255
                self.state = IntroSceneState.STATE_IDLE
                self.blink_increasing = True

        elif self.state == IntroSceneState.STATE_IDLE:
            if self.blink_increasing:
                self.blink_alpha += 1
                if self.blink_alpha >= 255:
                    self.blink_alpha = 255
                    self.blink_increasing = False
            else:
                self.blink_alpha -= 2
                if self.blink_alpha <= 2:
                    self.blink_alpha = 50
                    self.blink_increasing = True

        elif self.state == IntroSceneState.STATE_FADE_OUT:
            self.alpha -= FADE_SPEED
            if self.alpha <= 0:
                self.alpha = 0
                self.manager.change_scene(EntityScene(self.manager),
                                          transition_scene=TransitionScene)

    def render(self, screen):
        screen.fill(DARK_BLACK)

        # renderizo titulo
        self.title_surface.set_alpha(self.alpha)
        screen.blit(self.title_surface, self.text_rect)

        # press key
        if self.state == IntroSceneState.STATE_IDLE:
            self.press_key_surface.set_alpha(self.blink_alpha)
            screen.blit(self.press_key_surface, self.press_key_rect)
        elif self.state == IntroSceneState.STATE_FADE_OUT:
            self.press_key_surface.set_alpha(self.alpha)
            screen.blit(self.press_key_surface, self.press_key_rect)


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


if __name__ == "__main__":
    game = Game()
    game.run()
