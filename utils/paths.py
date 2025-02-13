import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BIOME_DIR = os.path.join(BASE_DIR, 'biome')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
GAME_DIR = os.path.join(BASE_DIR, 'game')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
ASSETS_DIR = os.path.join(GAME_DIR, 'assets')
RESEARCH_DIR = os.path.join(GAME_DIR, 'research')
UTILS_DIR = os.path.join(GAME_DIR, 'utils')
BIOME_DATA_DIR = os.path.join(BIOME_DIR, 'data')

