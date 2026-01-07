"""Phase 2 Guardrails & Face Routing Experiments (E2.1–E2.8).

Source of truth: `ECDD_Paper_DR_3_Experimentation.md` Phase 2 section.

Goal:
- Deterministic, parameterized guardrails that prevent nonsense inputs reaching the model
- Explicit reason codes
- Version-pinned face detector behavior

This file is intentionally separate from `phase2_experiments.py` so that:
- the original phase2 experiments remain unchanged
- Phase 2 *guardrails contract* can evolve independently and be used as a gate

Outputs:
- JSON results written to `PhaseWise_Experiments_ECDD/phase2_guardrails_results/`

Dependencies:
- `pip install mediapipe` (for backend='mediapipe')

"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Add parent directory to path for ecdd_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecdd_core.pipeline import (
    DecodeConfig,
    PreprocessConfig,
    FaceDetectorConfig,
    QualityGateConfig,
    decode_image_bytes,
    apply_guardrails,
)


@dataclass
class GuardrailExperimentResult:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]


def _write(out_dir: Path, result: GuardrailExperimentResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{result.experiment_id}.json").write_text(json.dumps(asdict(result), indent=2))


def _load_images(dir_path: Path, exts=(".jpg", ".png", ".webp"), limit: Optional[int] = None) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(dir_path.glob(f"*{ext}")))
    if limit is not None:
        files = files[:limit]
    return files


def _rgb_from_path(path: Path, decode_cfg: DecodeConfig) -> np.ndarray:
    data = path.read_bytes()
    return decode_image_bytes(data, decode_cfg)


def e2_1_face_detector_version_pin_test() -> GuardrailExperimentResult:
    """E2.1 Face detector version pin test.

    We record backend name + mediapipe version (if used) so runs are auditable.
    """
    import importlib

    cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=0.5)

    try:
        mp = importlib.import_module("mediapipe")
        mp_version = getattr(mp, "__version__", "unknown")
    except Exception as e:
        return GuardrailExperimentResult(
            experiment_id="E2.1",
            name="Face detector version pin test",
            passed=False,
            details={"error": "mediapipe not installed", "install": "pip install mediapipe"},
        )

    return GuardrailExperimentResult(
        experiment_id="E2.1",
        name="Face detector version pin test",
        passed=True,
        details={
            "backend": cfg.backend,
            "mediapipe_version": mp_version,
            "min_confidence": cfg.min_confidence,
            "multi_face_policy": cfg.multi_face_policy,
        },
    )


def e2_2_no_face_abstain_correctness_test(base_data_dir: Path) -> GuardrailExperimentResult:
    """E2.2 No-face abstain correctness test on OOD set.

    Pass condition (strict): 100% of OOD images should NOT pass guardrails.
    """
    decode_cfg = DecodeConfig()
    face_cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=0.5, multi_face_policy="max")
    q_cfg = QualityGateConfig()

    ood_dir = base_data_dir / "ood"
    ood_imgs = _load_images(ood_dir, limit=None)
    if not ood_imgs:
        return GuardrailExperimentResult(
            experiment_id="E2.2",
            name="No-face abstain correctness test",
            passed=False,
            details={"error": f"No OOD images found in {ood_dir}"},
        )

    passed_count = 0
    failures = []
    for p in ood_imgs:
        rgb = _rgb_from_path(p, decode_cfg)
        gr = apply_guardrails(rgb, face_cfg, q_cfg)
        if gr.ok:
            passed_count += 1
            failures.append(p.name)

    passed = passed_count == 0
    return GuardrailExperimentResult(
        experiment_id="E2.2",
        name="No-face abstain correctness test",
        passed=passed,
        details={
            "total": len(ood_imgs),
            "passed_guardrails": passed_count,
            "failed_examples": failures[:10],
            "pass_condition": "0 OOD images may pass guardrails",
        },
    )


def e2_3_face_confidence_threshold_sweep(base_data_dir: Path, thresholds: Optional[List[float]] = None) -> GuardrailExperimentResult:
    """E2.3 Face confidence threshold sweep.

    Measures:
    - in-scope pass-through rate
    - OOD false pass-through rate

    This is a policy definition experiment; it does not declare pass/fail.
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.5, 0.7, 0.9]

    decode_cfg = DecodeConfig()
    q_cfg = QualityGateConfig()

    faces_dir = base_data_dir / "real"
    ood_dir = base_data_dir / "ood"

    face_imgs = _load_images(faces_dir, limit=20)
    ood_imgs = _load_images(ood_dir, limit=20)

    if not face_imgs or not ood_imgs:
        return GuardrailExperimentResult(
            experiment_id="E2.3",
            name="Face confidence threshold sweep",
            passed=False,
            details={"error": "Need both real/ and ood/ images"},
        )

    sweep = []
    for t in thresholds:
        face_cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=float(t), multi_face_policy="max")

        in_ok = 0
        for p in face_imgs:
            rgb = _rgb_from_path(p, decode_cfg)
            if apply_guardrails(rgb, face_cfg, q_cfg).ok:
                in_ok += 1

        ood_ok = 0
        for p in ood_imgs:
            rgb = _rgb_from_path(p, decode_cfg)
            if apply_guardrails(rgb, face_cfg, q_cfg).ok:
                ood_ok += 1

        sweep.append(
            {
                "min_confidence": t,
                "in_scope_pass_rate": in_ok / len(face_imgs),
                "ood_false_pass_rate": ood_ok / len(ood_imgs),
            }
        )

    return GuardrailExperimentResult(
        experiment_id="E2.3",
        name="Face confidence threshold sweep",
        passed=True,
        details={"sweep": sweep, "note": "Choose threshold that keeps OOD pass-through low without excessive in-scope abstain"},
    )


def e2_7_blur_metric_threshold_sweep(base_data_dir: Path, thresholds: Optional[List[float]] = None) -> GuardrailExperimentResult:
    """E2.7 Blur metric threshold sweep (variance of Laplacian).

    Uses edge case blur images if available.
    """
    if thresholds is None:
        thresholds = [20.0, 50.0, 100.0, 150.0]

    decode_cfg = DecodeConfig()
    face_cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=0.5)

    # Use edge_cases blur images if available
    edge_dir = base_data_dir / "edge_cases"
    blur_imgs = [p for p in _load_images(edge_dir, limit=None) if "blur" in p.stem.lower()]

    if not blur_imgs:
        return GuardrailExperimentResult(
            experiment_id="E2.7",
            name="Blur metric threshold sweep",
            passed=False,
            details={"error": "No blur edge case images found (filenames containing 'blur')"},
        )

    sweep = []
    for t in thresholds:
        q_cfg = QualityGateConfig(blur_laplacian_var_threshold=float(t))
        ok = 0
        for p in blur_imgs:
            rgb = _rgb_from_path(p, decode_cfg)
            if apply_guardrails(rgb, face_cfg, q_cfg).ok:
                ok += 1
        sweep.append({"blur_threshold": t, "pass_rate": ok / len(blur_imgs), "n": len(blur_imgs)})

    return GuardrailExperimentResult(
        experiment_id="E2.7",
        name="Blur metric threshold sweep",
        passed=True,
        details={"sweep": sweep},
    )


def e2_8_compression_proxy_threshold_sweep(base_data_dir: Path, thresholds: Optional[List[float]] = None) -> GuardrailExperimentResult:
    """E2.8 Compression proxy sweep.

    Uses edge case JPEG quality images if available.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4]

    decode_cfg = DecodeConfig()
    face_cfg = FaceDetectorConfig(backend="mediapipe", min_confidence=0.5)

    edge_dir = base_data_dir / "edge_cases"
    jpeg_imgs = [p for p in _load_images(edge_dir, limit=None) if "jpeg_quality" in p.stem.lower()]

    if not jpeg_imgs:
        return GuardrailExperimentResult(
            experiment_id="E2.8",
            name="Compression proxy threshold sweep",
            passed=False,
            details={"error": "No JPEG quality edge case images found (filenames containing 'jpeg_quality')"},
        )

    sweep = []
    for t in thresholds:
        q_cfg = QualityGateConfig(enable_compression_proxy=True, min_jpeg_quality_proxy=float(t))
        ok = 0
        for p in jpeg_imgs:
            rgb = _rgb_from_path(p, decode_cfg)
            if apply_guardrails(rgb, face_cfg, q_cfg).ok:
                ok += 1
        sweep.append({"compression_proxy_threshold": t, "pass_rate": ok / len(jpeg_imgs), "n": len(jpeg_imgs)})

    return GuardrailExperimentResult(
        experiment_id="E2.8",
        name="Compression proxy threshold sweep",
        passed=True,
        details={"sweep": sweep, "note": "Proxy is heuristic; prefer future JPEG quant-table estimator"},
    )


def run_all_phase2_guardrails(out_dir: Optional[Path] = None) -> Dict[str, GuardrailExperimentResult]:
    out_dir = out_dir or (Path(__file__).parent / "phase2_guardrails_results")
    base_data_dir = Path(__file__).resolve().parents[1] / "ECDD_Experiment_Data"

    results = {
        "E2.1": e2_1_face_detector_version_pin_test(),
        "E2.2": e2_2_no_face_abstain_correctness_test(base_data_dir),
        "E2.3": e2_3_face_confidence_threshold_sweep(base_data_dir),
        # TODO (next): E2.4 alignment/crop determinism
        # TODO (next): E2.5 multi-face policy A/B
        # TODO (next): E2.6 min face size sweep
        "E2.7": e2_7_blur_metric_threshold_sweep(base_data_dir),
        "E2.8": e2_8_compression_proxy_threshold_sweep(base_data_dir),
    }

    for r in results.values():
        _write(out_dir, r)

    return results


if __name__ == "__main__":
    out = Path(__file__).parent / "phase2_guardrails_results"
    run_all_phase2_guardrails(out)
    print(f"Wrote Phase 2 guardrails results to {out}")
