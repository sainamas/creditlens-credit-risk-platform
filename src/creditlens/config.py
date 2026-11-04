
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model storage
MODEL_DIR = PROJECT_ROOT / "models"

# Reproducibility
RANDOM_STATE = 42

# Machine learning target
TARGET_COLUMN = "default payment next month"
