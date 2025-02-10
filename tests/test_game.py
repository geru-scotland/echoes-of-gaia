import os
from unittest.mock import patch
from game.main import init_systems, Game

# Para evitar cuando lo ponga en GH actions
# que falle por no disponer de video mode.
if os.getenv("CI"):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

@patch.object(Game, 'run', side_effect=lambda: None)
def test_game_runs(mock_run):
    try:
        init_systems()
        game = Game()
        game.run()
        assert True
    except Exception as e:
        assert False, f"Game failed to start: {e}"