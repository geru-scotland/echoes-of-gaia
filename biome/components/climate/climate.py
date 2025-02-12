import simpy


class Climate:
    def __init__(self, env: simpy.Environment):
        self._env = env
        self._env.process(self._update(25))
        self._env.process(self._evolve(100))

    # Cambios drásticos de clima, deberían de ser dispatcheados mejor, esto para probar solo.
    def _update(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            print(f"Updating Climate: t={self._env.now}")
            yield self._env.timeout(25)

    def _evolve(self, delay: int):
        yield self._env.timeout(delay)
        while True:
            print(f"Evolving... t={self._env.now}")
            yield self._env.timeout(100)

