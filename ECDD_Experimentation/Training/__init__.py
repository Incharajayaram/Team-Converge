"""ECDD Training Package.

Contains:
- models/: Model architectures (LaDeDa ResNet50)
- training/: Training scripts and utilities
- data/: Dataset preparation and augmentation
"""

from pathlib import Path

TRAINING_DIR = Path(__file__).parent
MODELS_DIR = TRAINING_DIR / "models"
DATA_DIR = TRAINING_DIR / "data"
