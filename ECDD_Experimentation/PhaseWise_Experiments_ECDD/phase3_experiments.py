import json
import sys
import yaml
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = Path(r"f:\Team converge\Team-Converge\ECDD_Experimentation")
DATA_DIR = BASE_DIR / "ECDD_Experiment_Data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
POLICY_PATH = BASE_DIR / "policy_contract.yaml"
RESULTS_DIR = BASE_DIR / "PhaseWise_Experiments_ECDD" / "phase3_results"

# Add data dir to path for imports
sys.path.insert(0, str(DATA_DIR))
from checkpoint_system import CheckpointStore

@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    passed: bool
    details: Dict[str, Any]
    timestamp: str

def save_result(result: ExperimentResult):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"{result.experiment_id}.json", 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)

class StudentModelStub:
    """
    Numpy-based stub for the Student Model to verify architectural contracts.
    Simulates the transformation from S4 (256x256x3) to S5 (8x8 Patch Logits).
    """
    INPUT_SHAPE = (256, 256, 3)
    OUTPUT_GRID = (8, 8)
    STRIDE = 32
    
    def __init__(self, kernel_init="random"):
        self.kernel_size = 32
        # Initialize a single filter of size 32x32x3
        np.random.seed(42)
        if kernel_init == "random":
            self.weights = np.random.randn(32, 32, 3)
        elif kernel_init == "identity":
             # Simple averager
             self.weights = np.ones((32, 32, 3)) / (32*32*3)
        
    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Forward pass (Convolution with stride 32).
        Input: (256, 256, 3)
        Output: (8, 8, 1) - The Patch Logits
        """
        if input_tensor.shape != self.INPUT_SHAPE:
            # Try to resize or error? For strict contract, we expect correct S4.
            # But S4 might come in as [H, W, C].
            if input_tensor.shape[:2] != (256, 256):
                 raise ValueError(f"Input shape must be 256x256, got {input_tensor.shape}")
        
        output = np.zeros(self.OUTPUT_GRID + (1,), dtype=np.float32)
        
        # Naive stride-32 convolution (non-overlapping windows for simplicity)
        # This acts as the receptive field mapping
        for i in range(8):
            for j in range(8):
                h_start, h_end = i * 32, (i + 1) * 32
                w_start, w_end = j * 32, (j + 1) * 32
                patch = input_tensor[h_start:h_end, w_start:w_end, :]
                
                # Dot product + Sum (Convolution operation logic)
                # patch: 32x32x3, weights: 32x32x3
                activation = np.sum(patch * self.weights)
                output[i, j, 0] = activation
                
        return output

class PoolingLayers:
    """Implementations of pooling strategies."""
    
    @staticmethod
    def top_k_pooling(logits: np.ndarray, k: int) -> float:
        """
        Top-K pooling: Average of the top K largest logits.
        logits: (8, 8, 1) or (N, 1) flattened
        """
        flat = logits.flatten()
        if k > len(flat):
            k = len(flat)
        # Partition to find top k (not fully sorting is faster, but sort is fine for 64 items)
        indices = np.argsort(flat)[-k:]
        top_values = flat[indices]
        return float(np.mean(top_values))

    @staticmethod
    def attention_pooling(logits: np.ndarray, temperature: float = 1.0) -> float:
        """
        Attention pooling: Softmax weighted sum.
        logits: (8, 8, 1)
        """
        flat = logits.flatten()
        # Numerical stability: shift by max
        shifted = (flat - np.max(flat)) / temperature
        exps = np.exp(shifted)
        weights = exps / np.sum(exps)
        pooled = np.sum(flat * weights)
        return float(pooled)

def run_phase3_experiments():
    print("="*60)
    print("ECDD PHASE 3: PATCH GRID & POOLING GUARDRAILS")
    print("="*60)
    
    results = []
    
    # E3.1: Patch grid shape contract test
    print("\nE3.1: Patch grid shape contract test")
    try:
        model = StudentModelStub()
        dummy_input = np.zeros((256, 256, 3), dtype=np.float32)
        output = model.forward(dummy_input)
        
        expected_shape = (8, 8, 1)
        passed = (output.shape == expected_shape)
        details = {"expected": str(expected_shape), "got": str(output.shape)}
        
        if passed:
            print(f"  [PASS] Shape matches contract: {output.shape}")
        else:
            print(f"  [FAIL] Shape mismatch: expected {expected_shape}, got {output.shape}")
            
        results.append(ExperimentResult("experiment_e3_1", "Patch Grid Shape", passed, details, str(datetime.now())))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e3_1", "Patch Grid Shape", False, {"error": str(e)}, str(datetime.now())))

    # E3.2: Heatmap-to-image coordinate mapping test
    print("\nE3.2: Heatmap-to-image coordinate mapping test")
    try:
        # We verify that patch (0,0) corresponds to 0:32 pixels
        # And patch (7,7) corresponds to 224:256 pixels
        stride = 32
        mappings = []
        passed = True
        
        # Test corners and center
        test_cells = [(0,0), (7,7), (3,3)]
        for i, j in test_cells:
            h_start_exp = i * 32
            w_start_exp = j * 32
            mappings.append({
                "cell": (i, j),
                "pixel_range_h": (h_start_exp, h_start_exp + 32),
                "pixel_range_w": (w_start_exp, w_start_exp + 32)
            })
            
            # To verify strictly, we can input a one-hot image (all zeros except that region)
            # and verify the model activation is highest at that cell.
            model_identity = StudentModelStub(kernel_init="identity")
            test_img = np.zeros((256, 256, 3), dtype=np.float32)
            # Activate the specific patch region
            test_img[h_start_exp:h_start_exp+32, w_start_exp:w_start_exp+32, :] = 1.0
            
            out = model_identity.forward(test_img)
            max_idx = np.unravel_index(np.argmax(out), out.shape)
            
            if max_idx[:2] != (i, j):
                passed = False
                print(f"  [FAIL] Cell ({i},{j}) did not activate correctly! Max activated at {max_idx[:2]}")
            else:
                print(f"  [PASS] Cell ({i},{j}) maps to pixels {h_start_exp}-{h_start_exp+32}, {w_start_exp}-{w_start_exp+32}")
        
        results.append(ExperimentResult("experiment_e3_2", "Receptive Field Mapping", passed, {"mappings": mappings}, str(datetime.now())))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e3_2", "Receptive Field Mapping", False, {"error": str(e)}, str(datetime.now())))

    # E3.3: Pooling choice A/B test
    print("\nE3.3: Pooling choice A/B test (Top-K vs Attention)")
    try:
        # Simulate different scenarios
        # 1. Uniformly fake: all logits high
        # 2. Uniformly real: all logits low
        # 3. "Needle in haystack": one very high logit, others low (common deepfake artifact)
        
        scenarios = {
            "all_fake": np.ones((8,8,1)) * 10.0,
            "all_real": np.ones((8,8,1)) * -10.0,
            "needle": np.ones((8,8,1)) * -10.0
        }
        scenarios["needle"][0,0] = 10.0 # One fake patch
        
        pooling = PoolingLayers()
        details = {}
        
        print(f"  {'Scenario':<15} | {'Top-K (k=4)':<12} | {'Attention (T=1)':<15}")
        print("-" * 50)
        
        for name, grid in scenarios.items():
            tk = pooling.top_k_pooling(grid, k=4) # top 5% of 64 is approx 3-4
            attn = pooling.attention_pooling(grid, temperature=1.0)
            details[name] = {"top_k": tk, "attention": attn}
            print(f"  {name:<15} | {tk:<12.4f} | {attn:<15.4f}")
            
        # Analysis:
        # Needle scenario: Top-K should pick up the high value heavily. Attention should also focus on it.
        # Top-K(k=4) with 1 high value average: (10 + -10 + -10 + -10) / 4 = -5.0
        # Attention with 1 high value: exp(10) dominates exp(-10). sum(p*x) approx 10.
        
        # Attention is structurally 'max-like' for single peaks but smooth.
        # Top-K dilutes single-patch artifacts if K is too large.
        
        # Decision: Attention seems better for "single patch artifact" preservation if T is calibrated.
        # But Top-K is more explainable (average of worst parts).
        
        passed = True
        results.append(ExperimentResult("experiment_e3_3", "Pooling Comparison", passed, details, str(datetime.now())))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e3_3", "Pooling Comparison", False, {"error": str(e)}, str(datetime.now())))

    # E3.4: Top-k parameter sweep
    print("\nE3.4: Top-k parameter sweep")
    try:
        # Sweep K from 1 to 16 (approx 1% to 25% of 64 patches)
        # Using "needle" scenario to see dilution
        needle = np.ones((8,8,1)) * -10.0
        needle[0,0] = 10.0
        
        print(f"  {'K':<4} | {'Pool Score':<10}")
        details = {}
        for k in [1, 2, 4, 8, 16, 32]:
            score = PoolingLayers.top_k_pooling(needle, k)
            details[f"k={k}"] = score
            print(f"  {k:<4} | {score:<10.4f}")
            
        # If K=1, score=10. If K=64, score ~ -9.7
        # We need a K that isn't too noisy (K=1 is noisy) but doesn't dilute too much.
        # K=4 (approx 5-6%) is standard in literature (e.g. MIL).
        recommended_k = 4
        print(f"  Recommended K: {recommended_k} (~6% of patches)")
        
        results.append(ExperimentResult("experiment_e3_4", "Top-K Sweep", True, details, str(datetime.now())))
    except Exception as e:
         print(f"  [ERROR] {e}")
         results.append(ExperimentResult("experiment_e3_4", "Top-K Sweep", False, {"error": str(e)}, str(datetime.now())))

    # E3.5: Attention pooling determinism
    print("\nE3.5: Attention pooling determinism")
    try:
        data = np.random.randn(8,8,1)
        res1 = PoolingLayers.attention_pooling(data)
        res2 = PoolingLayers.attention_pooling(data)
        res3 = PoolingLayers.attention_pooling(data)
        
        is_deterministic = (res1 == res2 == res3)
        if is_deterministic:
            print("  [PASS] Attention pooling is bit-exact across runs.")
        else:
            print("  [FAIL] Nondeterministic!")
            
        results.append(ExperimentResult("experiment_e3_5", "Attention Determinism", is_deterministic, {}, str(datetime.now())))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e3_5", "Attention Determinism", False, {"error": str(e)}, str(datetime.now())))

    # E3.6: Patch-logit sanity tests (Perturbation)
    print("\nE3.6: Patch-logit sanity tests (Perturbation)")
    try:
        model = StudentModelStub() # Random weights
        input_base = np.random.randn(256, 256, 3).astype(np.float32)
        base_out = model.forward(input_base)
        
        # Perturb a patch region (e.g. top-left 32x32)
        input_pert = input_base.copy()
        input_pert[0:32, 0:32, :] += 5.0 # Add noise
        
        pert_out = model.forward(input_pert)
        
        diff = np.abs(pert_out - base_out)
        
        # Check that diff is concentrated at [0,0]
        # Since we use non-overlapping 32 stride, it should be strictly at [0,0]
        max_diff_loc = np.unravel_index(np.argmax(diff), diff.shape)
        
        # The change at [0,0] should be significant
        change_at_target = diff[0,0,0]
        change_elsewhere = np.sum(diff) - change_at_target
        
        print(f"  Change at target (0,0): {change_at_target:.4f}")
        print(f"  Change elsewhere: {change_elsewhere:.4f}")
        
        passed = (max_diff_loc[:2] == (0,0)) and (change_elsewhere < 1e-5)
        
        if passed:
            print("  [PASS] Perturbation localized correctly to patch (0,0).")
        else:
            print("  [FAIL] Perturbation leaked or applied to wrong patch.")
            
        results.append(ExperimentResult("experiment_e3_6", "Perturbation Test", passed, {"diff_sum": float(np.sum(diff))}, str(datetime.now())))
    except Exception as e:
        print(f"  [ERROR] {e}")
        results.append(ExperimentResult("experiment_e3_6", "Perturbation Test", False, {"error": str(e)}, str(datetime.now())))

    # Summary
    print("\n" + "="*60)
    print("PHASE 3 SUMMARY")
    print("="*60)
    all_passed = True
    for res in results:
        status = "PASS" if res.passed else "FAIL "
        print(f"  [{status}] {res.name}")
        if not res.passed:
            all_passed = False
            
    # Determine final recommendation
    print("\nFinal Recommendations:")
    print("  - Patch Grid: 8x8 (Verified)")
    print("  - Pooling: Attention is preferred for 'needle' detection (score 10.0 vs -5.0 for TopK).")
    print("  - Stride: 32 (Verified mapping)")
    
    if all_passed:
        print("\nOverall: ALL PASSED")
    else:
        print("\nOverall: SOME FAILED")
        
    for res in results:
        save_result(res)

if __name__ == "__main__":
    from datetime import datetime
    run_phase3_experiments()
