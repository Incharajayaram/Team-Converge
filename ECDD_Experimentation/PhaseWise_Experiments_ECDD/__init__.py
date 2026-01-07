"""ECDD PhaseWise Experiments Package.

Contains all phase experiments (E1-E8) for ECDD validation.

Phases:
- Phase 1: Pixel pipeline equivalence (E1.1-E1.9)
- Phase 2: Face detection and guardrails (E2.1-E2.8)
- Phase 3: Patch grid and pooling (E3.1-E3.6)
- Phase 4: Calibration and thresholds (E4.1-E4.6)
- Phase 5: Quantization and parity (E5.1-E5.5)
- Phase 6: Evaluation battery (E6.1-E6.4)
- Phase 7: Monitoring and drift (E7.1-E7.3)
- Phase 8: Dataset governance (E8.1-E8.3)
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR  # Results saved alongside experiments
