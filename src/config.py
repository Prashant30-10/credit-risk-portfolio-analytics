from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

REFERENCE_DATE = "2026-10-27"
RANDOM_STATE = 42

