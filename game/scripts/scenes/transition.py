# Scene manager haré instancia en game
from game.systems.scenes.base_scene import Scene


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