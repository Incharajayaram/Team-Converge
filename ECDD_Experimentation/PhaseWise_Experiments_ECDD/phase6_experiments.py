"""Phase 6 Experiments: Realistic evaluation battery.

Implements E6.1–E6.4 from `ECDD_Paper_DR_3_Experimentation.md`.

This is a scaffold that will be wired to ecdd_core evaluation battery modules
(source/time splits, transform suite, OOD separation reporting).

Outputs should be written to:
  PhaseWise_Experiments_ECDD/phase6_results/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for ecdd_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


@dataclass
class Phase6Result:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]


def _write_result(out_dir: Path, result: Phase6Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{result.experiment_id}.json").write_text(json.dumps(asdict(result), indent=2))


def e6_1_source_based_split_stress_test(
    metadata_json: Optional[Path] = None,
    holdout_sources: Optional[list] = None,
) -> Phase6Result:
    """E6.1: Source-based split stress test.
    
    Re-split data by generator/source/device/platform/compression rather than random.
    Evaluates whether model performance is stable across different data sources.
    
    Args:
        metadata_json: JSON file with sample metadata (id, label, path, source, timestamp)
        holdout_sources: List of source names to hold out for testing
    """
    from ecdd_core.eval.splits import source_based_split, SampleMeta
    
    if metadata_json is None:
        # Mock mode: generate synthetic metadata
        np.random.seed(46)
        sources = ["generator_A", "generator_B", "device_X", "device_Y", "platform_Z"]
        samples = []
        for i in range(100):
            samples.append(SampleMeta(
                id=f"sample_{i:03d}",
                label=np.random.randint(0, 2),
                path=f"/mock/path/{i:03d}.jpg",
                source=np.random.choice(sources),
                timestamp=f"2024-01-{(i%28)+1:02d}T00:00:00",
            ))
        is_mock = True
    else:
        return Phase6Result(
            experiment_id="E6.1",
            name="Source-based split stress test",
            passed=False,
            details={"error": "Real metadata loading not yet implemented."},
        )
    
    if holdout_sources is None:
        # Use one source as holdout
        unique_sources = list(set(s.source for s in samples))
        holdout_sources = [unique_sources[0]] if unique_sources else []
    
    train, test = source_based_split(samples, holdout_sources)
    
    # Compute distribution stats
    train_label_dist = {0: 0, 1: 0}
    test_label_dist = {0: 0, 1: 0}
    for s in train:
        train_label_dist[s.label] += 1
    for s in test:
        test_label_dist[s.label] += 1
    
    # Check if split is reasonable
    has_train = len(train) > 0
    has_test = len(test) > 0
    test_has_both_classes = test_label_dist[0] > 0 and test_label_dist[1] > 0
    
    passed = has_train and has_test and test_has_both_classes
    
    return Phase6Result(
        experiment_id="E6.1",
        name="Source-based split stress test",
        passed=passed,
        details={
            "is_mock": is_mock,
            "holdout_sources": holdout_sources,
            "num_train": len(train),
            "num_test": len(test),
            "train_label_dist": train_label_dist,
            "test_label_dist": test_label_dist,
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real metadata and model for performance evaluation.",
            "recommendation": "Evaluate model on test split to measure cross-source generalization.",
        },
    )


def e6_2_time_based_split_drift_probe(
    metadata_json: Optional[Path] = None,
    split_timestamp: Optional[str] = None,
) -> Phase6Result:
    """E6.2: Time-based split drift probe.
    
    Sort data by collection time; train/calibrate on earlier, test on later.
    Simulates real deployment where future data may differ from training distribution.
    
    Args:
        metadata_json: JSON file with sample metadata
        split_timestamp: ISO timestamp to split train/test (earlier = train, later = test)
    """
    from ecdd_core.eval.splits import time_based_split, SampleMeta
    
    if metadata_json is None:
        # Mock mode: generate temporal metadata
        np.random.seed(47)
        samples = []
        for i in range(100):
            samples.append(SampleMeta(
                id=f"sample_{i:03d}",
                label=np.random.randint(0, 2),
                path=f"/mock/path/{i:03d}.jpg",
                source="mock_source",
                timestamp=f"2024-{(i//30)+1:02d}-{(i%30)+1:02d}T00:00:00",
            ))
        is_mock = True
    else:
        return Phase6Result(
            experiment_id="E6.2",
            name="Time-based split drift probe",
            passed=False,
            details={"error": "Real metadata loading not yet implemented."},
        )
    
    if split_timestamp is None:
        # Use median timestamp as split
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)
        split_timestamp = sorted_samples[len(sorted_samples) // 2].timestamp
    
    train, test = time_based_split(samples, split_timestamp)
    
    # Distribution stats
    train_label_dist = {0: 0, 1: 0}
    test_label_dist = {0: 0, 1: 0}
    for s in train:
        train_label_dist[s.label] += 1
    for s in test:
        test_label_dist[s.label] += 1
    
    has_train = len(train) > 0
    has_test = len(test) > 0
    test_has_both_classes = test_label_dist[0] > 0 and test_label_dist[1] > 0
    
    passed = has_train and has_test and test_has_both_classes
    
    return Phase6Result(
        experiment_id="E6.2",
        name="Time-based split drift probe",
        passed=passed,
        details={
            "is_mock": is_mock,
            "split_timestamp": split_timestamp,
            "num_train": len(train),
            "num_test": len(test),
            "train_label_dist": train_label_dist,
            "test_label_dist": test_label_dist,
            "train_time_range": f"{train[0].timestamp} to {train[-1].timestamp}" if train else "N/A",
            "test_time_range": f"{test[0].timestamp} to {test[-1].timestamp}" if test else "N/A",
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real temporal metadata and model.",
            "recommendation": "Monitor for temporal drift; recalibrate if performance degrades on recent data.",
        },
    )


def e6_3_transform_suite_conclusive_test(golden_faces_dir: Optional[Path] = None) -> Phase6Result:
    """E6.3: Apply deterministic transform suite to a small face set.

    This is a *battery harness*; it does not assume a specific model yet.
    It validates that the transform suite is executable and produces stable outputs.

    Later we will plug in model scoring and operating-point metrics.
    """
    from ecdd_core.eval import default_transform_suite, apply_transform
    from PIL import Image

    # Default to ECDD_Experiment_Data/real for now
    if golden_faces_dir is None:
        golden_faces_dir = Path(__file__).resolve().parents[1] / "ECDD_Experiment_Data" / "real"

    imgs = sorted(list(golden_faces_dir.glob("*.jpg")) + list(golden_faces_dir.glob("*.png")))[:10]
    if not imgs:
        return Phase6Result(
            experiment_id="E6.3",
            name="Transform suite conclusive test",
            passed=False,
            details={"error": f"No images found in {golden_faces_dir}"},
        )

    suite = default_transform_suite()
    produced = []
    for p in imgs:
        base = Image.open(p).convert("RGB")
        for spec in suite:
            out = apply_transform(base, spec)
            produced.append({"image": p.name, "transform": spec.name, "params": spec.params, "size": out.size})

    return Phase6Result(
        experiment_id="E6.3",
        name="Transform suite conclusive test",
        passed=True,
        details={"num_inputs": len(imgs), "num_transforms": len(suite), "outputs": produced[:25]},
    )


def e6_4_out_of_scope_separation_test(
    ood_dir: Optional[Path] = None,
    face_detector_backend: str = "stub",
) -> Phase6Result:
    """E6.4: Out-of-scope separation test.
    
    Ensure OOD images (animals, scenery, cartoons, documents) trigger abstention
    or are labeled out-of-scope. These should be excluded from face accuracy claims.
    
    Args:
        ood_dir: Directory containing OOD images
        face_detector_backend: Face detector to use ('stub' or 'mediapipe')
    """
    from ecdd_core.pipeline.face import detect_faces, FaceDetectorConfig
    from PIL import Image
    import numpy as np
    
    if ood_dir is None:
        # Use default OOD directory
        ood_dir = Path(__file__).resolve().parents[1] / "ECDD_Experiment_Data" / "ood"
    
    if not ood_dir.exists():
        return Phase6Result(
            experiment_id="E6.4",
            name="Out-of-scope separation test",
            passed=False,
            details={"error": f"OOD directory not found: {ood_dir}"},
        )
    
    ood_images = sorted(list(ood_dir.glob("*.jpg")) + list(ood_dir.glob("*.png")))
    
    if len(ood_images) == 0:
        return Phase6Result(
            experiment_id="E6.4",
            name="Out-of-scope separation test",
            passed=False,
            details={"error": f"No OOD images found in {ood_dir}"},
        )
    
    cfg = FaceDetectorConfig(backend=face_detector_backend, min_confidence=0.5)
    
    results = []
    abstain_count = 0
    
    for img_path in ood_images[:20]:  # Test first 20 images
        try:
            img = Image.open(img_path).convert("RGB")
            rgb = np.array(img, dtype=np.uint8)
            detection = detect_faces(rgb, cfg)
            
            # OOD images should have no faces detected (=> abstain)
            num_faces = len(detection.boxes)
            should_abstain = num_faces == 0
            
            if should_abstain:
                abstain_count += 1
            
            results.append({
                "image": img_path.name,
                "num_faces": num_faces,
                "should_abstain": should_abstain,
            })
        except Exception as e:
            results.append({
                "image": img_path.name,
                "error": str(e),
            })
    
    # Pass if at least 80% of OOD images abstain (no faces detected)
    abstain_rate = abstain_count / len(results) if results else 0.0
    passed = abstain_rate >= 0.8
    
    return Phase6Result(
        experiment_id="E6.4",
        name="Out-of-scope separation test",
        passed=passed,
        details={
            "ood_dir": str(ood_dir),
            "num_tested": len(results),
            "abstain_count": abstain_count,
            "abstain_rate": abstain_rate,
            "face_detector": face_detector_backend,
            "results": results[:10],  # Show first 10
            "status": "PASS" if passed else "FAIL",
            "recommendation": "OOD samples should always abstain or be excluded from face accuracy metrics.",
            "note": "Abstention via no-face-detection. With real model, also check explicit OOD detection.",
        },
    )


def run_all_phase6(out_dir: Optional[Path] = None) -> Dict[str, Phase6Result]:
    out_dir = out_dir or (Path(__file__).parent / "phase6_results")
    results = {
        "E6.1": e6_1_source_based_split_stress_test(),
        "E6.2": e6_2_time_based_split_drift_probe(),
        "E6.3": e6_3_transform_suite_conclusive_test(),
        "E6.4": e6_4_out_of_scope_separation_test(),
    }
    for r in results.values():
        _write_result(out_dir, r)
    return results


if __name__ == "__main__":
    out = Path(__file__).parent / "phase6_results"
    run_all_phase6(out)
    print(f"Wrote Phase 6 scaffold results to {out}")
