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
class DependencyInjector:
    def __init__(self):
        self._dependencies = {}

    def register(self, name, instance):
        if name in self._dependencies.keys():
            raise KeyError(f"Dependency '{name}' already registered in the injector.")
        self._dependencies[name] = instance

    def get(self, name):
        if name not in self._dependencies:
            raise KeyError(f"Dependency '{name}' not found in the injector.")
        return self._dependencies[name]


dependency_injector = DependencyInjector()
