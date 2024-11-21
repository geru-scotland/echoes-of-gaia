from random import random


# enums



# functions

def generate_matte_color():
    return random.choice([
        (102, 102, 153),  # Bluish gray
        (102, 153, 102),  # Matte green
        (153, 102, 102),  # Matte red
        (153, 153, 102),  # Matte yellow
        (102, 102, 102)  # Dark matte gray
    ])