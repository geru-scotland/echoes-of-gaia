import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

GAME_DIR = os.path.join(BASE_DIR, 'game')
ASSETS_DIR = os.path.join(GAME_DIR, 'assets')

LOGS_DIR = os.path.join(BASE_DIR, 'logs')
