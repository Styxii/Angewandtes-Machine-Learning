"""Zentrale Konfiguration"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
