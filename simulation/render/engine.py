from logging import Logger
from typing import Dict

import pygame

from config.settings import RenderSettings
from simulation.render.components import Component


class RenderEngine:
    def __init__(self, settings: RenderSettings):
        self._initialized = False
        self._settings = settings
        self._logger: Logger = self._settings.get_logger("render")
        self._components: Dict[str, Component] = {}
        self._screen = None

    def is_initialized(self) -> bool:
        return self._initialized

    def init(self):
        self._logger.info("Initialising Rendering Engine")
        pygame.init()
        self._screen = pygame.display.set_mode((self._settings.window_width,
                                               self._settings.window_height))
        pygame.display.set_caption(self._settings.title)
        self._initialized = True
        self.run()

    def run(self):
        running: bool = True
        while running:
            for name, component in self._components.items():
                component.render(self._screen)
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    break
            pygame.display.flip()
        pygame.quit()

    def add_component(self, component: Component):
        if component.name not in self._components:
            self._components[component.name] = component

    @property
    def settings(self) -> RenderSettings:
        return self._settings




