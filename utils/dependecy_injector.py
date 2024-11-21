class DependencyInjector:
    def __init__(self):
        self._dependencies = {}

    def register(self, name, instance):
        self._dependencies[name] = instance

    def get(self, name):
        if name not in self._dependencies:
            raise KeyError(f"Dependency '{name}' not found in the injector.")
        return self._dependencies[name]

dependency_injector = DependencyInjector()