"""Phase 5 Experiments: Quantization and float-to-TFLite parity.

Implements E5.1–E5.5 from `ECDD_Paper_DR_3_Experimentation.md`.

NOTE: These experiments require actual float and TFLite models.
Since models may not be trained yet, we provide mock implementations that:
1. Test the harness infrastructure
2. Define parity tolerances
3. Can be re-run with real models when available

Outputs should be written to:
  PhaseWise_Experiments_ECDD/phase5_results/
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Add parent directory to path for ecdd_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


@dataclass
class Phase5Result:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]


def _json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_serializable(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_result(out_dir: Path, result: Phase5Result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _json_serializable(asdict(result))
    (out_dir / f"{result.experiment_id}.json").write_text(json.dumps(data, indent=2))


def _mock_model_outputs(n_samples: int = 10, add_noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mock float and TFLite model outputs for testing.
    
    Returns:
        float_outputs, tflite_outputs (with optional noise)
    """
    np.random.seed(42)
    float_outputs = np.random.randn(n_samples).astype(np.float32)
    tflite_outputs = float_outputs + np.random.randn(n_samples).astype(np.float32) * add_noise
    return float_outputs, tflite_outputs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def e5_1_float_vs_tflite_probability_parity(
    epsilon_prob: float = 0.05,
    golden_dir: Optional[Path] = None,
    float_model_path: Optional[Path] = None,
    tflite_model_path: Optional[Path] = None,
) -> Phase5Result:
    """E5.1: Float vs TFLite probability parity test (end-to-end).
    
    Compares final calibrated probabilities between float and TFLite models.
    
    Args:
        epsilon_prob: Maximum allowed absolute difference in probabilities
        golden_dir: Directory containing golden test images
        float_model_path: Path to float model (e.g., .pth, .onnx)
        tflite_model_path: Path to TFLite model
    """
    if float_model_path is None or tflite_model_path is None:
        # Mock mode: test the harness
        float_logits, tflite_logits = _mock_model_outputs(n_samples=20, add_noise=0.01)
        float_probs = _sigmoid(float_logits)
        tflite_probs = _sigmoid(tflite_logits)
        is_mock = True
    else:
        # Real mode: load models and run inference
        import sys
        import torch
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ecdd_core.export.tflite_converter import run_tflite_inference
        from Training.models.ladeda_resnet import create_ladeda_model
        
        # Load golden images
        if golden_dir is None:
            golden_dir = Path(__file__).parent.parent / "ECDD_Experiment_Data" / "real"
        
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images.extend(list(golden_dir.glob(ext))[:10])
        
        if not images:
            return Phase5Result(
                experiment_id="E5.1",
                name="Float vs TFLite probability parity test",
                passed=False,
                details={"error": f"No images found in {golden_dir}"},
            )
        
        # Preprocess images
        from PIL import Image, ImageOps
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        preprocessed = []
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - mean) / std
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            preprocessed.append(arr)
        
        input_data = np.stack(preprocessed, axis=0).astype(np.float32)
        
        # Run float model
        model = create_ladeda_model(pretrained=False)
        checkpoint = torch.load(float_model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data)
            float_logits_tensor, _, _ = model(input_tensor)
            float_logits = float_logits_tensor.numpy().squeeze()
        
        # Run TFLite model
        tflite_outputs = run_tflite_inference(tflite_model_path, input_data)
        tflite_logits = tflite_outputs[0].squeeze()
        
        float_probs = _sigmoid(float_logits)
        tflite_probs = _sigmoid(tflite_logits)
        is_mock = False
    
    # Compute parity metrics
    max_abs_diff = float(np.max(np.abs(float_probs - tflite_probs)))
    mean_abs_diff = float(np.mean(np.abs(float_probs - tflite_probs)))
    
    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(float_probs, tflite_probs)
    
    passed = max_abs_diff <= epsilon_prob and rank_corr >= 0.95
    
    return Phase5Result(
        experiment_id="E5.1",
        name="Float vs TFLite probability parity test",
        passed=passed,
        details={
            "is_mock": is_mock,
            "epsilon_prob": epsilon_prob,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rank_correlation": float(rank_corr),
            "num_samples": len(float_probs),
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real models for production validation.",
        },
    )


def e5_2_patch_logit_map_parity(
    epsilon_patch: float = 0.1,
    float_model_path: Optional[Path] = None,
    tflite_model_path: Optional[Path] = None,
) -> Phase5Result:
    """E5.2: Patch-logit map parity test (intermediate).
    
    Compares patch-logit maps between float and TFLite:
    - Shape equality
    - Mean/max absolute difference
    - Argmax location stability (where highest fake patch is)
    """
    if float_model_path is None or tflite_model_path is None:
        # Mock mode: simulate patch-logit maps
        np.random.seed(43)
        h, w = 8, 8  # 8x8 patch grid
        float_map = np.random.randn(h, w).astype(np.float32)
        tflite_map = float_map + np.random.randn(h, w).astype(np.float32) * 0.05
        is_mock = True
    else:
        return Phase5Result(
            experiment_id="E5.2",
            name="Patch-logit map parity test",
            passed=False,
            details={"error": "Real model inference not yet implemented."},
        )
    
    # Shape check
    shape_equal = float_map.shape == tflite_map.shape
    
    # Difference metrics
    max_abs_diff = float(np.max(np.abs(float_map - tflite_map)))
    mean_abs_diff = float(np.mean(np.abs(float_map - tflite_map)))
    
    # Argmax stability
    float_argmax = np.unravel_index(np.argmax(float_map), float_map.shape)
    tflite_argmax = np.unravel_index(np.argmax(tflite_map), tflite_map.shape)
    argmax_stable = float_argmax == tflite_argmax
    
    passed = shape_equal and max_abs_diff <= epsilon_patch and argmax_stable
    
    return Phase5Result(
        experiment_id="E5.2",
        name="Patch-logit map parity test",
        passed=passed,
        details={
            "is_mock": is_mock,
            "epsilon_patch": epsilon_patch,
            "shape_equal": shape_equal,
            "float_shape": list(float_map.shape),
            "tflite_shape": list(tflite_map.shape),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "argmax_stable": argmax_stable,
            "float_argmax": list(float_argmax),
            "tflite_argmax": list(tflite_argmax),
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real models for production validation.",
        },
    )


def e5_3_pooled_logit_parity(
    epsilon_pooled: float = 0.05,
    float_model_path: Optional[Path] = None,
    tflite_model_path: Optional[Path] = None,
) -> Phase5Result:
    """E5.3: Pooled logit parity test.
    
    Compares pooled logit (single scalar per image) before calibration.
    """
    if float_model_path is None or tflite_model_path is None:
        # Mock mode
        float_logits, tflite_logits = _mock_model_outputs(n_samples=20, add_noise=0.02)
        is_mock = True
    else:
        return Phase5Result(
            experiment_id="E5.3",
            name="Pooled logit parity test",
            passed=False,
            details={"error": "Real model inference not yet implemented."},
        )
    
    max_abs_diff = float(np.max(np.abs(float_logits - tflite_logits)))
    mean_abs_diff = float(np.mean(np.abs(float_logits - tflite_logits)))
    
    passed = max_abs_diff <= epsilon_pooled
    
    return Phase5Result(
        experiment_id="E5.3",
        name="Pooled logit parity test",
        passed=passed,
        details={
            "is_mock": is_mock,
            "epsilon_pooled": epsilon_pooled,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "num_samples": len(float_logits),
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real models for production validation.",
        },
    )


def e5_4_post_quant_calibration_gate(
    calibration_json: Optional[Path] = None,
    tflite_model_path: Optional[Path] = None,
) -> Phase5Result:
    """E5.4: Post-quant calibration mandatory gate.
    
    Refit calibration parameters using quantized model logits on calibration set.
    Recompute thresholds and verify operating point is restored.
    """
    if tflite_model_path is None:
        # Mock mode: simulate pre/post calibration
        np.random.seed(44)
        # Simulate quantized logits (slightly shifted from float)
        float_logits = np.random.randn(100).astype(np.float32)
        quant_logits = float_logits + np.random.randn(100).astype(np.float32) * 0.1
        labels = (float_logits > 0).astype(int)
        
        # Simulate calibration improvement
        from ecdd_core.calibration.temperature_scaling import fit_temperature, apply_temperature, sigmoid, expected_calibration_error
        
        pre_probs = sigmoid(quant_logits)
        pre_ece = expected_calibration_error(pre_probs, labels)
        
        params, _ = fit_temperature(quant_logits, labels)
        post_logits = apply_temperature(quant_logits, params)
        post_probs = sigmoid(post_logits)
        post_ece = expected_calibration_error(post_probs, labels)
        
        improved = post_ece <= pre_ece
        is_mock = True
    else:
        return Phase5Result(
            experiment_id="E5.4",
            name="Post-quant calibration mandatory gate",
            passed=False,
            details={"error": "Real model inference not yet implemented."},
        )
    
    passed = improved
    
    return Phase5Result(
        experiment_id="E5.4",
        name="Post-quant calibration mandatory gate",
        passed=passed,
        details={
            "is_mock": is_mock,
            "pre_ece": float(pre_ece),
            "post_ece": float(post_ece),
            "temperature": float(params.temperature),
            "improved": improved,
            "status": "PASS" if passed else "FAIL",
            "note": "Mock data used. Re-run with real quantized model for production validation.",
        },
    )


def e5_5_delegate_and_threading_invariance(
    tflite_model_path: Optional[Path] = None,
    delegates: Optional[list] = None,
    thread_counts: Optional[list] = None,
) -> Phase5Result:
    """E5.5: Delegate and threading invariance test.
    
    Run TFLite with different delegates (XNNPACK, NNAPI, GPU) and thread counts.
    Verify outputs remain within tolerances.
    """
    if delegates is None:
        delegates = ["default", "xnnpack"]
    if thread_counts is None:
        thread_counts = [1, 2, 4]
    
    if tflite_model_path is None:
        # Mock mode: simulate slight variations across configs
        np.random.seed(45)
        baseline = np.random.randn(10).astype(np.float32)
        
        configs = []
        max_diff_vs_baseline = 0.0
        
        for delegate in delegates:
            for threads in thread_counts:
                # Simulate tiny variations
                output = baseline + np.random.randn(10).astype(np.float32) * 0.001
                diff = float(np.max(np.abs(output - baseline)))
                max_diff_vs_baseline = max(max_diff_vs_baseline, diff)
                configs.append({
                    "delegate": delegate,
                    "threads": threads,
                    "max_diff_vs_baseline": diff,
                })
        
        tolerance = 0.01
        passed = max_diff_vs_baseline <= tolerance
        is_mock = True
    else:
        return Phase5Result(
            experiment_id="E5.5",
            name="Delegate and threading invariance test",
            passed=False,
            details={"error": "Real TFLite runtime testing not yet implemented."},
        )
    
    return Phase5Result(
        experiment_id="E5.5",
        name="Delegate and threading invariance test",
        passed=passed,
        details={
            "is_mock": is_mock,
            "configs_tested": configs,
            "max_diff_vs_baseline": max_diff_vs_baseline,
            "tolerance": tolerance,
            "status": "PASS" if passed else "FAIL",
            "recommendation": "Pin runtime config in policy_contract.yaml",
            "note": "Mock data used. Re-run with real TFLite model for production validation.",
        },
    )


def run_all_phase5(out_dir: Optional[Path] = None) -> Dict[str, Phase5Result]:
    out_dir = out_dir or (Path(__file__).parent / "phase5_results")
    results = {
        "E5.1": e5_1_float_vs_tflite_probability_parity(),
        "E5.2": e5_2_patch_logit_map_parity(),
        "E5.3": e5_3_pooled_logit_parity(),
        "E5.4": e5_4_post_quant_calibration_gate(),
        "E5.5": e5_5_delegate_and_threading_invariance(),
    }
    for r in results.values():
        _write_result(out_dir, r)
    return results


if __name__ == "__main__":
    out = Path(__file__).parent / "phase5_results"
    run_all_phase5(out)
    print(f"Wrote Phase 5 scaffold results to {out}")
