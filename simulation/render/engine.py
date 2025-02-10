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

    def is_initialized(self) -> bool:
        return self._initialized

    def init(self):
        pygame.init()
        self._initialized = True
        self._logger.info("Initialising Rendering Engine")

    def add_component(self, component: Component):
        if component.name not in self._components:
            self._components[component.name] = component

    @property
    def settings(self) -> RenderSettings:
        return self._settings




