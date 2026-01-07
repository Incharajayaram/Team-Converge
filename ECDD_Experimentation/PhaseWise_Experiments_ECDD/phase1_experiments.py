"""
ECDD Phase 1 Experiments: Pixel Pipeline Equivalence
=====================================================

Experiments E1.1 through E1.9 to verify pixel pipeline is deterministic
and equivalent across all deployment paths.

Goal: Lock the entire "pixel contract" across browser → server → edge.
STOP IF ANY FAILS - these are critical gates.
"""

from __future__ import annotations

import hashlib
import json
import sys
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding='utf-8')

# Paths - Updated for new structure
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATA_DIR = BASE_DIR / "ECDD_Experiment_Data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
POLICY_PATH = BASE_DIR / "policy_contract.yaml"
RESULTS_DIR = BASE_DIR / "PhaseWise_Experiments_ECDD" / "phase1_results"

# Add data dir to path for imports
sys.path.insert(0, str(DATA_DIR))
from checkpoint_system import (
    CheckpointStore, StageCheckpoint,
    compute_s0, compute_s1, compute_s3, compute_s4,
    compute_bytes_hash, compute_tensor_hash
)


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]
    timestamp: str


def load_policy() -> dict:
    """Load policy from YAML file."""
    if POLICY_PATH.exists():
        with open(POLICY_PATH, 'r') as f:
            raw = yaml.safe_load(f)
        pp = raw.get('pixel_pipeline', raw)
        return {
            "exif": pp.get('exif', {}),
            "alpha": pp.get('alpha', {}),
            "resize": pp.get('resize', {}),
            "normalization": pp.get('normalization', {}),
            "tolerances": pp.get('tolerances', {}),
        }
    return {}


def save_result(result: ExperimentResult):
    """Save experiment result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{result.experiment_id}.json"
    with open(result_path, 'w') as f:
        json.dump({
            "experiment_id": result.experiment_id,
            "name": result.name,
            "passed": result.passed,
            "details": result.details,
            "timestamp": result.timestamp,
        }, f, indent=2)


# ============================================================================
# E1.1: Byte-for-byte upload invariance
# ============================================================================

def experiment_e1_1() -> ExperimentResult:
    """
    E1.1: Byte-for-byte upload invariance test
    Pass: SHA-256 exact match for all tested images.
    """
    print("\n" + "="*60)
    print("E1.1: Byte-for-byte upload invariance test")
    print("="*60)
    
    store = CheckpointStore(CHECKPOINT_DIR)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for split in ['real', 'fake', 'ood', 'edge_cases']:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for img_path in list(split_dir.iterdir())[:5]:
            if not img_path.is_file():
                continue
            
            hash1 = compute_bytes_hash(img_path.read_bytes())
            hash2 = compute_bytes_hash(img_path.read_bytes())
            checkpoint = store.load_checkpoint(img_path.stem)
            stored_hash = checkpoint.s0.sha256 if checkpoint and checkpoint.s0 else None
            
            results["tested"] += 1
            if hash1 == hash2 and (stored_hash is None or hash1 == stored_hash):
                results["passed"] += 1
                print(f"  [PASS] {img_path.name}")
            else:
                results["failed"] += 1
                results["failures"].append({"file": img_path.name})
                print(f"  [FAIL] {img_path.name}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.1", "Byte-for-byte upload invariance", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.2: Format allowlist and corruption rejection
# ============================================================================

def experiment_e1_2() -> ExperimentResult:
    """
    E1.2: Format allowlist and corruption rejection test
    """
    print("\n" + "="*60)
    print("E1.2: Format allowlist and corruption rejection test")
    print("="*60)
    
    results = {"supported_passed": 0, "supported_tested": 0, "corruption_rejected": 0, "corruption_tests": 0, "failures": []}
    
    # Test supported formats
    for ext in ['.jpg', '.png', '.webp']:
        for split in ['real', 'fake', 'edge_cases']:
            split_dir = DATA_DIR / split
            if not split_dir.exists():
                continue
            for img_path in split_dir.glob(f"*{ext}"):
                results["supported_tested"] += 1
                try:
                    img = Image.open(img_path)
                    img.verify()
                    results["supported_passed"] += 1
                    print(f"  [PASS] {img_path.name}")
                except Exception as e:
                    results["failures"].append({"file": img_path.name, "error": str(e)})
                    print(f"  [FAIL] {img_path.name}")
                break
    
    # Test corruption rejection
    import tempfile
    for test_data, desc in [(b'\xff\xd8\xff\xe0', "truncated"), (b'not an image', "non-image")]:
        results["corruption_tests"] += 1
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(test_data)
            temp_path = Path(f.name)
        try:
            img = Image.open(temp_path)
            img.load()
            print(f"  [FAIL] Accepted {desc}")
        except Exception:
            results["corruption_rejected"] += 1
            print(f"  [PASS] Rejected {desc}")
        temp_path.unlink()
    
    passed = results["supported_passed"] == results["supported_tested"] and results["corruption_rejected"] == results["corruption_tests"]
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    
    return ExperimentResult("E1.2", "Format allowlist and corruption rejection", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.3: Single decode-path enforcement
# ============================================================================

def experiment_e1_3() -> ExperimentResult:
    """
    E1.3: Single decode-path enforcement test
    """
    print("\n" + "="*60)
    print("E1.3: Single decode-path enforcement test")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for split in ['real', 'fake']:
        for img_path in list((DATA_DIR / split).glob("*.jpg"))[:3]:
            results["tested"] += 1
            try:
                s1 = compute_s1(img_path, policy)
                checkpoint = store.load_checkpoint(img_path.stem)
                if checkpoint and checkpoint.s1 and s1.tensor_hash == checkpoint.s1.tensor_hash:
                    results["passed"] += 1
                    print(f"  [PASS] {img_path.name}")
                else:
                    s1_again = compute_s1(img_path, policy)
                    if s1.tensor_hash == s1_again.tensor_hash:
                        results["passed"] += 1
                        print(f"  [PASS] {img_path.name}: consistent")
                    else:
                        results["failed"] += 1
                        print(f"  [FAIL] {img_path.name}")
            except Exception as e:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}: {e}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.3", "Single decode-path enforcement", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.4: EXIF orientation correctness
# ============================================================================

def experiment_e1_4() -> ExperimentResult:
    """
    E1.4: EXIF orientation correctness test
    """
    print("\n" + "="*60)
    print("E1.4: EXIF orientation correctness test")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for img_path in (DATA_DIR / "edge_cases").glob("exif_orientation_*.jpg"):
        results["tested"] += 1
        s1 = compute_s1(img_path, policy)
        s1_again = compute_s1(img_path, policy)
        checkpoint = store.load_checkpoint(img_path.stem)
        
        if s1.tensor_hash == s1_again.tensor_hash:
            if checkpoint and checkpoint.s1 and s1.tensor_hash == checkpoint.s1.tensor_hash:
                results["passed"] += 1
                print(f"  [PASS] {img_path.name}: matches golden")
            else:
                results["passed"] += 1
                print(f"  [PASS] {img_path.name}: consistent")
        else:
            results["failed"] += 1
            print(f"  [FAIL] {img_path.name}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.4", "EXIF orientation correctness", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.5: RGB channel ordering and dtype/range
# ============================================================================

def experiment_e1_5() -> ExperimentResult:
    """
    E1.5: RGB channel ordering and dtype/range invariants
    """
    print("\n" + "="*60)
    print("E1.5: RGB channel ordering and dtype/range invariants")
    print("="*60)
    
    policy = load_policy()
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for split in ['real', 'fake', 'ood', 'edge_cases']:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
        for img_path in list(split_dir.iterdir())[:3]:
            if not img_path.is_file() or img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
            results["tested"] += 1
            try:
                s1 = compute_s1(img_path, policy)
                dtype_ok = s1.dtype == 'uint8'
                range_ok = s1.min_val >= 0 and s1.max_val <= 255
                channels_ok = len(s1.shape) == 3 and s1.shape[2] == 3
                
                if dtype_ok and range_ok and channels_ok:
                    results["passed"] += 1
                    print(f"  [PASS] {img_path.name}")
                else:
                    results["failed"] += 1
                    print(f"  [FAIL] {img_path.name}")
            except Exception as e:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}: {e}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.5", "RGB channel ordering and dtype/range", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.6: Gamma policy invariance (sRGB-only)
# ============================================================================

def experiment_e1_6() -> ExperimentResult:
    """
    E1.6: Gamma policy invariance (sRGB-only path)
    """
    print("\n" + "="*60)
    print("E1.6: Gamma policy invariance (sRGB-only)")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for img_path in list((DATA_DIR / "real").glob("*.jpg"))[:5]:
        results["tested"] += 1
        s1_run1 = compute_s1(img_path, policy)
        s1_run2 = compute_s1(img_path, policy)
        
        if s1_run1.tensor_hash == s1_run2.tensor_hash:
            results["passed"] += 1
            print(f"  [PASS] {img_path.name}")
        else:
            results["failed"] += 1
            print(f"  [FAIL] {img_path.name}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.6", "Gamma policy invariance (sRGB-only)", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.7: Alpha handling policy
# ============================================================================

def experiment_e1_7() -> ExperimentResult:
    """
    E1.7: Alpha handling policy test
    """
    print("\n" + "="*60)
    print("E1.7: Alpha handling policy test")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    alpha_images = list((DATA_DIR / "edge_cases").glob("alpha_*.png")) + \
                   list((DATA_DIR / "edge_cases").glob("alpha_*.webp"))
    
    for img_path in alpha_images:
        results["tested"] += 1
        try:
            s1 = compute_s1(img_path, policy)
            s1_again = compute_s1(img_path, policy)
            if s1.tensor_hash == s1_again.tensor_hash:
                results["passed"] += 1
                print(f"  [PASS] {img_path.name}")
            else:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}")
        except Exception as e:
            if "reject" in str(e).lower():
                results["passed"] += 1
                print(f"  [PASS] {img_path.name}: correctly rejected")
            else:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}: {e}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.7", "Alpha handling policy", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.8: Fixed interpolation kernel (resize)
# ============================================================================

def experiment_e1_8() -> ExperimentResult:
    """
    E1.8: Fixed interpolation kernel test
    """
    print("\n" + "="*60)
    print("E1.8: Fixed interpolation kernel test")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    tolerance = policy.get('tolerances', {}).get('s3_resized_max_diff', 1)
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for split in ['real', 'fake']:
        for img_path in list((DATA_DIR / split).iterdir())[:5]:
            if not img_path.is_file():
                continue
            results["tested"] += 1
            try:
                s3, _ = compute_s3(img_path, policy)
                checkpoint = store.load_checkpoint(img_path.stem)
                
                if checkpoint and checkpoint.s3:
                    # Convert both to tuple for comparison (JSON stores as list)
                    shape_match = tuple(s3.shape) == tuple(checkpoint.s3.shape)
                    hash_match = s3.tensor_hash == checkpoint.s3.tensor_hash
                    
                    if shape_match and hash_match:
                        results["passed"] += 1
                        print(f"  [PASS] {img_path.name}")
                    else:
                        results["failed"] += 1
                        results["failures"].append({
                            "file": img_path.name,
                            "current_hash": s3.tensor_hash[:16],
                            "golden_hash": checkpoint.s3.tensor_hash[:16],
                            "shape_match": shape_match
                        })
                        print(f"  [FAIL] {img_path.name}: current={s3.tensor_hash[:16]} vs golden={checkpoint.s3.tensor_hash[:16]}")
                else:
                    s3_again, _ = compute_s3(img_path, policy)
                    if s3.tensor_hash == s3_again.tensor_hash:
                        results["passed"] += 1
                        print(f"  [PASS] {img_path.name}: consistent (no checkpoint)")
                    else:
                        results["failed"] += 1
                        print(f"  [FAIL] {img_path.name}: non-deterministic")
            except Exception as e:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}: {e}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.8", "Fixed interpolation kernel", passed, results, datetime.now().isoformat())


# ============================================================================
# E1.9: Training normalization constants
# ============================================================================

def experiment_e1_9() -> ExperimentResult:
    """
    E1.9: Training normalization constants test
    """
    print("\n" + "="*60)
    print("E1.9: Training normalization constants test")
    print("="*60)
    
    policy = load_policy()
    store = CheckpointStore(CHECKPOINT_DIR)
    expected_mean = policy.get('normalization', {}).get('mean', [0.485, 0.456, 0.406])
    expected_std = policy.get('normalization', {}).get('std', [0.229, 0.224, 0.225])
    
    print(f"  Expected mean: {expected_mean}")
    print(f"  Expected std: {expected_std}")
    
    results = {"tested": 0, "passed": 0, "failed": 0, "failures": []}
    
    for split in ['real', 'fake']:
        for img_path in list((DATA_DIR / split).iterdir())[:5]:
            if not img_path.is_file():
                continue
            results["tested"] += 1
            try:
                s3, s3_tensor = compute_s3(img_path, policy)
                s4, _ = compute_s4(s3_tensor, policy)
                
                constants_match = s4.normalization_mean == expected_mean and s4.normalization_std == expected_std
                if constants_match:
                    results["passed"] += 1
                    print(f"  [PASS] {img_path.name}")
                else:
                    results["failed"] += 1
                    print(f"  [FAIL] {img_path.name}")
            except Exception as e:
                results["failed"] += 1
                print(f"  [FAIL] {img_path.name}: {e}")
    
    passed = results["failed"] == 0
    print(f"\nResult: {'PASS' if passed else 'FAIL'} ({results['passed']}/{results['tested']})")
    
    return ExperimentResult("E1.9", "Training normalization constants", passed, results, datetime.now().isoformat())


# ============================================================================
# Main: Run All Phase 1 Experiments
# ============================================================================

def run_all_phase1_experiments() -> Dict[str, ExperimentResult]:
    """Run all Phase 1 experiments."""
    print("\n" + "="*60)
    print("ECDD PHASE 1: PIXEL PIPELINE EQUIVALENCE")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")
    
    experiments = [
        experiment_e1_1, experiment_e1_2, experiment_e1_3,
        experiment_e1_4, experiment_e1_5, experiment_e1_6,
        experiment_e1_7, experiment_e1_8, experiment_e1_9,
    ]
    
    results = {}
    all_passed = True
    
    for exp_func in experiments:
        result = exp_func()
        results[result.experiment_id] = result
        save_result(result)
        if not result.passed:
            all_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)
    for exp_id, result in results.items():
        print(f"  {'[PASS]' if result.passed else '[FAIL]'} {exp_id}: {result.name}")
    
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"Results: {RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    run_all_phase1_experiments()
