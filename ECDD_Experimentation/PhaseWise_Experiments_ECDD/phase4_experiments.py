"""Phase 4 Experiments: Calibration, thresholds, and abstain semantics.

Implements E4.1–E4.6 from `ECDD_Paper_DR_3_Experimentation.md`.

This is a scaffold that will be wired to ecdd_core calibration modules.
Partner can begin preparing calibration sets and running the harness once
calibration utilities are finalized.

Outputs should be written to:
  PhaseWise_Experiments_ECDD/phase4_results/

NOTE: This file lives under ECDD_Experimentation and does not modify deepfake-patch-audit.
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
class Phase4Result:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]


def _write_result(out_dir: Path, result: Phase4Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{result.experiment_id}.json").write_text(json.dumps(asdict(result), indent=2))


def e4_1_calibration_set_contract_test(calibration_json: Optional[Path] = None, min_size: int = 500) -> Phase4Result:
    """E4.1: Define and validate calibration set contract.

    Validates:
    - Calibration set exists and meets minimum size
    - Balanced distribution (not heavily skewed)
    - Contains required metadata fields
    """
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    
    if calibration_json is None:
        return Phase4Result(
            experiment_id="E4.1",
            name="Calibration set contract test",
            passed=False,
            details={"error": "No calibration set provided. Pass --calibration_json path."},
        )
    
    if not calibration_json.exists():
        return Phase4Result(
            experiment_id="E4.1",
            name="Calibration set contract test",
            passed=False,
            details={"error": f"Calibration file not found: {calibration_json}"},
        )
    
    try:
        logits, labels, rows = load_calibration_json(calibration_json)
        info = describe_calibration_set(logits, labels, calibration_json)
        
        # Validation checks
        checks = {}
        checks["size_ok"] = len(logits) >= min_size
        checks["has_both_classes"] = info.num_real > 0 and info.num_fake > 0
        
        # Balance check: neither class should be < 20% of total
        balance_ratio = min(info.num_real, info.num_fake) / max(info.num_real, info.num_fake)
        checks["balanced"] = balance_ratio >= 0.2
        
        # All rows have required fields
        checks["has_ids"] = all("id" in r for r in rows)
        
        passed = all(checks.values())
        
        return Phase4Result(
            experiment_id="E4.1",
            name="Calibration set contract test",
            passed=passed,
            details={
                "calibration_set": asdict(info),
                "checks": checks,
                "min_size_requirement": min_size,
                "balance_ratio": float(balance_ratio),
            },
        )
    except Exception as e:
        return Phase4Result(
            experiment_id="E4.1",
            name="Calibration set contract test",
            passed=False,
            details={"error": str(e)},
        )


def _require_calibration_file(path: Optional[Path]) -> Path:
    if path is None:
        raise ValueError(
            "Calibration logits file required. Provide --calibration_json pointing to a JSON list of {id,logit,label}."
        )
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def e4_2_temperature_scaling_fit_and_verify(calibration_json: Optional[Path] = None) -> Phase4Result:
    """E4.2: Fit temperature scaling and verify calibration improvement.

    Input: calibration JSON (see ecdd_core.calibration.calibration_set_contract).
    Output: fitted temperature, pre/post ECE, pre/post NLL.
    """
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    from ecdd_core.calibration.temperature_scaling import fit_temperature, apply_temperature, sigmoid, nll_from_logits, expected_calibration_error

    p = _require_calibration_file(calibration_json)
    logits, labels, _rows = load_calibration_json(p)

    info = describe_calibration_set(logits, labels, p)

    # Before
    pre_probs = sigmoid(logits)
    pre_nll = nll_from_logits(logits, labels)
    pre_ece = expected_calibration_error(pre_probs, labels)

    params, fit_details = fit_temperature(logits, labels)
    scaled_logits = apply_temperature(logits, params)

    post_probs = sigmoid(scaled_logits)
    post_nll = nll_from_logits(scaled_logits, labels)
    post_ece = expected_calibration_error(post_probs, labels)

    passed = post_ece <= pre_ece  # should not worsen calibration

    return Phase4Result(
        experiment_id="E4.2",
        name="Temperature scaling fit and verification",
        passed=passed,
        details={
            "calibration_set": asdict(info),
            "temperature": params.temperature,
            "pre": {"nll": pre_nll, "ece": pre_ece},
            "post": {"nll": post_nll, "ece": post_ece},
            "fit": fit_details,
        },
    )


def e4_3_platt_scaling_fit_and_verify(calibration_json: Optional[Path] = None) -> Phase4Result:
    """E4.3: Fit Platt scaling (a,b) and verify calibration improvement."""
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    from ecdd_core.calibration.platt_scaling import fit_platt, apply_platt
    from ecdd_core.calibration.temperature_scaling import expected_calibration_error

    p = _require_calibration_file(calibration_json)
    logits, labels, _rows = load_calibration_json(p)
    info = describe_calibration_set(logits, labels, p)

    # Pre (uncalibrated)
    from ecdd_core.calibration.temperature_scaling import sigmoid

    pre_probs = sigmoid(logits)
    pre_ece = expected_calibration_error(pre_probs, labels)

    params, fit_details = fit_platt(logits, labels)
    post_probs = apply_platt(logits, params)
    post_ece = expected_calibration_error(post_probs, labels)

    passed = post_ece <= pre_ece

    return Phase4Result(
        experiment_id="E4.3",
        name="Platt scaling fit and verification",
        passed=passed,
        details={
            "calibration_set": asdict(info),
            "params": {"a": params.a, "b": params.b},
            "pre": {"ece": pre_ece},
            "post": {"ece": post_ece},
            "fit": fit_details,
        },
    )


def e4_4_operating_point_selection_fixed_error_budget(calibration_json: Optional[Path] = None, target_fpr: float = 0.05) -> Phase4Result:
    """E4.4: Select operating point threshold at fixed error budget (e.g., FPR<=5%)."""
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    from ecdd_core.calibration.temperature_scaling import sigmoid
    from ecdd_core.calibration.operating_point import select_threshold_at_fpr

    p = _require_calibration_file(calibration_json)
    logits, labels, _rows = load_calibration_json(p)
    info = describe_calibration_set(logits, labels, p)

    probs = sigmoid(logits)
    op, details = select_threshold_at_fpr(probs, labels, target_fpr=target_fpr)

    # Pass if we actually meet the target
    passed = details.get("selected_fpr", 1.0) <= target_fpr + 1e-6

    return Phase4Result(
        experiment_id="E4.4",
        name="Operating point selection test",
        passed=passed,
        details={
            "calibration_set": asdict(info),
            "operating_point": {"threshold": op.threshold, "target_fpr": op.target_fpr},
            "metrics": details,
        },
    )


def e4_5_abstain_band_sweep(calibration_json: Optional[Path] = None, target_fpr: float = 0.05, band_widths: Optional[list[float]] = None) -> Phase4Result:
    """E4.5: Design abstain band and sweep band width.

    We define:
      - t_fake: threshold meeting target FPR
      - t_real: t_fake - width
      - abstain if t_real <= p_fake < t_fake

    Reports error on non-abstained and abstain rate.
    """
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    from ecdd_core.calibration.temperature_scaling import sigmoid
    from ecdd_core.calibration.operating_point import select_threshold_at_fpr

    if band_widths is None:
        band_widths = [0.05, 0.1, 0.15, 0.2]

    p = _require_calibration_file(calibration_json)
    logits, labels, _rows = load_calibration_json(p)
    info = describe_calibration_set(logits, labels, p)

    probs = sigmoid(logits)
    op, _details = select_threshold_at_fpr(probs, labels, target_fpr=target_fpr)
    t_fake = op.threshold

    sweep = []
    for w in band_widths:
        t_real = max(0.0, t_fake - w)
        abstain = (probs >= t_real) & (probs < t_fake)
        preds = (probs >= t_fake).astype(int)

        # Only evaluate on non-abstained
        mask = ~abstain
        if mask.sum() == 0:
            err = None
        else:
            err = float(np.mean(preds[mask] != labels[mask]))

        sweep.append(
            {
                "band_width": w,
                "t_real": float(t_real),
                "t_fake": float(t_fake),
                "abstain_rate": float(np.mean(abstain)),
                "error_non_abstained": err,
            }
        )

    # Pass if at least one configuration reduces error on non-abstained vs baseline
    baseline_err = float(np.mean((probs >= t_fake).astype(int) != labels))
    best = min([s for s in sweep if s["error_non_abstained"] is not None], key=lambda s: s["error_non_abstained"], default=None)
    passed = best is not None and best["error_non_abstained"] <= baseline_err

    return Phase4Result(
        experiment_id="E4.5",
        name="Abstain band design sweep",
        passed=passed,
        details={
            "calibration_set": asdict(info),
            "target_fpr": target_fpr,
            "baseline_threshold": float(t_fake),
            "baseline_error": baseline_err,
            "sweep": sweep,
            "best": best,
        },
    )


def e4_6_guardrail_conditioned_threshold_policy(
    calibration_json: Optional[Path] = None, 
    quality_threshold: float = 0.5,
    stricter_multiplier: float = 1.2
) -> Phase4Result:
    """E4.6: Compare conditional policies on low-quality inputs (abstain vs stricter threshold).
    
    Simulates two policies:
    - Policy A: Force abstain on low-quality samples
    - Policy B: Apply stricter threshold (higher bar for declaring fake)
    
    Uses a mock quality score derived from logit variance as a proxy.
    """
    from ecdd_core.calibration.calibration_set_contract import load_calibration_json, describe_calibration_set
    from ecdd_core.calibration.temperature_scaling import sigmoid
    from ecdd_core.calibration.operating_point import select_threshold_at_fpr
    
    if calibration_json is None:
        return Phase4Result(
            experiment_id="E4.6",
            name="Guardrail-conditioned threshold policy test",
            passed=False,
            details={"error": "No calibration set provided"},
        )
    
    p = _require_calibration_file(calibration_json)
    logits, labels, _rows = load_calibration_json(p)
    info = describe_calibration_set(logits, labels, p)
    
    probs = sigmoid(logits)
    op, _details = select_threshold_at_fpr(probs, labels, target_fpr=0.05)
    base_threshold = op.threshold
    strict_threshold = min(0.99, base_threshold * stricter_multiplier)
    
    # Mock quality score: higher uncertainty = lower quality
    # Use distance from 0.5 as proxy for confidence
    quality_scores = 1.0 - 2.0 * np.abs(probs - 0.5)
    low_quality_mask = quality_scores < quality_threshold
    
    # Policy A: Abstain on low quality
    preds_a = (probs >= base_threshold).astype(int)
    preds_a[low_quality_mask] = -1  # abstain marker
    
    # Policy B: Stricter threshold on low quality
    preds_b = np.where(
        low_quality_mask,
        (probs >= strict_threshold).astype(int),
        (probs >= base_threshold).astype(int)
    )
    
    # Metrics on high quality only (for both)
    high_quality_mask = ~low_quality_mask
    
    if high_quality_mask.sum() > 0:
        acc_a_hq = float(np.mean(preds_a[high_quality_mask] == labels[high_quality_mask]))
        acc_b_hq = float(np.mean(preds_b[high_quality_mask] == labels[high_quality_mask]))
    else:
        acc_a_hq = acc_b_hq = None
    
    # Abstain rate
    abstain_rate_a = float(np.mean(preds_a == -1))
    abstain_rate_b = 0.0  # Policy B never abstains
    
    # Low quality error rates
    if low_quality_mask.sum() > 0:
        # Policy A abstains, so no errors
        err_a_lq = 0.0
        err_b_lq = float(np.mean(preds_b[low_quality_mask] != labels[low_quality_mask]))
    else:
        err_a_lq = err_b_lq = None
    
    passed = True  # Informational experiment; we report tradeoffs
    
    return Phase4Result(
        experiment_id="E4.6",
        name="Guardrail-conditioned threshold policy test",
        passed=passed,
        details={
            "calibration_set": asdict(info),
            "base_threshold": float(base_threshold),
            "strict_threshold": float(strict_threshold),
            "quality_threshold": quality_threshold,
            "low_quality_fraction": float(np.mean(low_quality_mask)),
            "policy_a_abstain": {
                "abstain_rate": abstain_rate_a,
                "accuracy_high_quality": acc_a_hq,
                "error_low_quality": err_a_lq,
            },
            "policy_b_stricter": {
                "abstain_rate": abstain_rate_b,
                "accuracy_high_quality": acc_b_hq,
                "error_low_quality": err_b_lq,
            },
            "recommendation": "Choose policy_a (abstain) if low-quality errors are unacceptable. Choose policy_b if some errors are tolerable to reduce abstain rate.",
        },
    )


def run_all_phase4(out_dir: Optional[Path] = None, calibration_json: Optional[Path] = None) -> Dict[str, Phase4Result]:
    out_dir = out_dir or (Path(__file__).parent / "phase4_results")
    results = {
        "E4.1": e4_1_calibration_set_contract_test(calibration_json=calibration_json),
        "E4.2": e4_2_temperature_scaling_fit_and_verify(calibration_json=calibration_json),
        "E4.3": e4_3_platt_scaling_fit_and_verify(calibration_json=calibration_json),
        "E4.4": e4_4_operating_point_selection_fixed_error_budget(calibration_json=calibration_json),
        "E4.5": e4_5_abstain_band_sweep(calibration_json=calibration_json),
        "E4.6": e4_6_guardrail_conditioned_threshold_policy(calibration_json=calibration_json),
    }
    for r in results.values():
        _write_result(out_dir, r)
    return results


if __name__ == "__main__":
    # Usage:
    #   python phase4_experiments.py /path/to/calibration_logits.json
    # where calibration_logits.json is a JSON list of {id, logit, label}
    import sys

    out = Path(__file__).parent / "phase4_results"
    cal = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_all_phase4(out, calibration_json=cal)
    print(f"Wrote Phase 4 results to {out}")
