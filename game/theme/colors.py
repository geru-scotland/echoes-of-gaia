from random import random


class Colors:
    class Background:
        DARK_BLACK = (0, 0, 0)
        MATT_BLACK = (30, 30, 30)
        BRIGHT_WHITE = (255, 255, 255)

    class Text:
        PRIMARY_TEXT = (255, 255, 255)
        SECONDARY_TEXT = (200, 200, 200)
        WARNING_TEXT = (255, 50, 50)

    class UI:
        PRIMARY_BLUE = (0, 120, 255)
        HIGHLIGHT_GREEN = (0, 255, 120)

    @staticmethod
    def generate_matte_color():
        return random.choice([
            (102, 102, 153),  # Bluish gray
            (102, 153, 102),  # Matte green
            (153, 102, 102),  # Matte red
            (153, 153, 102),  # Matte yellow
            (102, 102, 102)  # Dark matte gray
        ])
