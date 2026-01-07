"""ECDD Experiment Data Package.

Contains golden datasets, edge cases, and checkpoint system.
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent
EDGE_CASES_DIR = DATA_DIR / "edge_cases"
REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"
OOD_DIR = DATA_DIR / "ood"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
