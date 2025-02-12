from logging import Logger
from typing import Dict
import queue

import pygame

from config.settings import RenderSettings
from simulation.render.components import RenderComponent


class RenderEngine:
    def __init__(self, settings: RenderSettings):
        self._initialized = False
        self._settings = settings
        self._logger: Logger = self._settings.get_logger("render")
        self._components: Dict[str, RenderComponent] = {}
        self._screen = None
        self._task_queue = queue.Queue()

    def is_initialized(self) -> bool:
        return self._initialized

    def enqueue_task(self, task, *args, **kwargs):
        self._task_queue.put((task, args, kwargs))

    def init(self):
        self._logger.info("Initialising Rendering Engine")
        pygame.init()
        self._screen = pygame.display.set_mode((self._settings.window_width,
                                               self._settings.window_height))
        pygame.display.set_caption(self._settings.title)
        self._initialized = True
        self.run()
    def _process_task_queue(self):
        try:
            while True:
                task, args, kwargs = self._task_queue.get_nowait()
                task(*args, **kwargs)
        # en python, entornos multhithread, ojo, es para evitar problemas con .empty entre hilos.
        except queue.Empty:
            pass

    def run(self):
        running: bool = True
        while running:
            self._process_task_queue()

            # Render de componentes
            self._screen.fill((0, 0, 0))
            for name, component in self._components.items():
                component.render(self._screen)

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
            pygame.display.flip()
        pygame.quit()

    def add_component(self, component: RenderComponent):
        try:
            if component.name not in self._components:
                self._components[component.name] = component
        except Exception as e:
            self._logger.exception(f"There was an error deleting a component: {e}")

    def remove_component(self, name: str):
        try:
            if name in self._components:
                del self._components[name]
        except Exception as e:
            self._logger.exception(f"There was an error deleting a component: {e}")


    @property
    def settings(self) -> RenderSettings:
        return self._settings




