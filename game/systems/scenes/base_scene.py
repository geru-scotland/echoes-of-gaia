"""
##########################################################################
#                                                                        #
#                           ✦ ECHOES OF GAIA ✦                           #
#                                                                        #
#    Trabajo Fin de Grado (TFG)                                          #
#    Facultad de Ingeniería Informática - Donostia                       #
#    UPV/EHU - Euskal Herriko Unibertsitatea                             #
#                                                                        #
#    Área de Computación e Inteligencia Artificial                       #
#                                                                        #
#    Autor:  Aingeru García Blas                                         #
#    GitHub: https://github.com/geru-scotland                            #
#    Repo:   https://github.com/geru-scotland/echoes-of-gaia             #
#                                                                        #
##########################################################################
"""
import os.path
from abc import abstractmethod
from logging import Logger

from shared.strings import Loggers
from utils.dependecy_injector import dependency_injector
from utils.loggers import LoggerManager
from utils.paths import ASSETS_DIR


class Scene:
    def __init__(self, scene_name, args=None, **kwargs):
        try:
            self._scene_name = scene_name
            self._assets = {}
            self._setup()
        except Exception as e:
            print(f"Critical error: Missing dependency - {e}")

    def _setup(self):
        self.settings = dependency_injector.get("scene_settings")
        self._scene_data = self.settings.load_scene_data(self._scene_name)
        self._screen_width, self._screen_height = self.settings.screen_width, self.settings.screen_height
        self._logger: Logger = LoggerManager.get_logger(Loggers.SCENE)

        self._load_assets()
        self._build_scene()
        self._start()

    @abstractmethod
    def _load_assets(self):
        pass

    @abstractmethod
    def _build_scene(self):
        pass

    @abstractmethod
    def _start(self):
        pass

    @abstractmethod
    def handle_events(self, event):
        pass

    @abstractmethod
    def update(self, diff):
        pass

    @abstractmethod
    def render(self, screen):
        pass
