import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'atp_tennis.csv'
MODELS_DIR = BASE_DIR / 'models'

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Page Config
PAGE_TITLE = "Tennis ML Predictor"
PAGE_ICON = "🎾"
LAYOUT = "wide"
