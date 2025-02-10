import pygame

from biome.systems.maps.procedural_maps import Map


class Component:
    name = None

    def __init__(self, name: str):
        self.name = name

    def update(self):
        pass

    def render(self, screen):
        pass


class MapComponent(Component):
    def __init__(self, map: Map, width, height):
        super().__init__("map")
        self._map: Map = map
        self.map_surface = pygame.Surface((width, height))

    def render(self, screen):
        screen.blit(self.map_surface, (0, 0))
