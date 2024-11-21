from enum import Enum


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