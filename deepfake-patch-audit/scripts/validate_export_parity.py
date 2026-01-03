#!/usr/bin/env python3
"""
Export Parity Harness - Validate TFLite/ONNX exports against PyTorch.

This script verifies that exported models produce outputs numerically close
to the original PyTorch model. Critical for deployment confidence.

Checks:
1. Logit distribution (mean, std, range)
2. Decision agreement at calibrated threshold
3. AUC delta between PyTorch and exported model
4. Per-sample output correlation

Usage:
    python scripts/validate_export_parity.py \
        --pytorch-checkpoint weights/student/student_best.pth \
        --tflite-path exports/student.tflite \
        --data-root data/ \
        --threshold 0.5 \
        --device cuda
"""

import sys
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.pooling import TopKLogitPooling
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader


# =============================================================================
# TFLite Inference Runner
# =============================================================================

class TFLiteRunner:
    """Run inference on TFLite model."""
    
    def __init__(self, tflite_path: str):
        """
        Args:
            tflite_path: Path to TFLite model
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required for TFLite inference. Install with: pip install tensorflow")
        
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Log model info
        print(f"TFLite Input:  {self.input_details[0]['shape']} ({self.input_details[0]['dtype']})")
        print(f"TFLite Output: {self.output_details[0]['shape']} ({self.output_details[0]['dtype']})")
    
    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        Run inference on input array.
        
        Args:
            input_array: Input array matching model's expected shape
        
        Returns:
            Output array
        """
        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_array)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output
    
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        """
        Run inference on a batch of images.
        
        Note: TFLite typically processes one image at a time.
        """
        images_np = images.cpu().numpy()
        outputs = []
        
        for i in range(images_np.shape[0]):
            # TFLite expects (1, H, W, C) or (1, C, H, W) depending on model
            img = images_np[i:i+1]
            
            # Check if channels need to be moved (PyTorch is NCHW, TF is NHWC)
            if self.input_details[0]['shape'][-1] == 3:  # NHWC format
                img = np.transpose(img, (0, 2, 3, 1))
            
            # Ensure correct dtype
            if self.input_details[0]['dtype'] == np.float32:
                img = img.astype(np.float32)
            
            output = self.predict(img)
            outputs.append(output)
        
        return np.concatenate(outputs, axis=0)


# =============================================================================
# ONNX Inference Runner
# =============================================================================

class ONNXRunner:
    """Run inference on ONNX model."""
    
    def __init__(self, onnx_path: str):
        """
        Args:
            onnx_path: Path to ONNX model
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime is required. Install with: pip install onnxruntime")
        
        # Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"ONNX Input:  {self.input_name} {self.input_shape}")
        print(f"ONNX Output: {self.output_name}")
    
    def predict_batch(self, images: torch.Tensor) -> np.ndarray:
        """Run inference on a batch of images."""
        images_np = images.cpu().numpy().astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: images_np})
        return outputs[0]


# =============================================================================
# Parity Validation
# =============================================================================

class ParityValidator:
    """
    Validate parity between PyTorch and exported models.
    """
    
    def __init__(
        self,
        pytorch_model,
        pooling,
        exported_runner,
        export_type: str = "tflite",
        device: str = "cuda",
    ):
        """
        Args:
            pytorch_model: PyTorch model
            pooling: TopK pooling layer
            exported_runner: TFLiteRunner or ONNXRunner
            export_type: "tflite" or "onnx"
            device: Device for PyTorch inference
        """
        self.pytorch_model = pytorch_model.to(device)
        self.pooling = pooling.to(device)
        self.exported_runner = exported_runner
        self.export_type = export_type
        self.device = device
        
        self.pytorch_model.eval()
    
    def validate_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict:
        """
        Compare PyTorch and exported model outputs for a batch.
        
        Returns:
            Dict with comparison metrics
        """
        # PyTorch inference
        with torch.no_grad():
            pytorch_patches = self.pytorch_model(images.to(self.device))
            pytorch_pooled = self.pooling(pytorch_patches)
            pytorch_probs = torch.sigmoid(pytorch_pooled.squeeze(-1)).cpu().numpy()
        
        # Exported model inference
        exported_output = self.exported_runner.predict_batch(images)
        
        # Handle different output formats
        if len(exported_output.shape) == 4:  # Patch output (B, 1, H, W)
            # Need to apply pooling equivalent
            # For simplicity, use mean of top-k values
            flat = exported_output.reshape(exported_output.shape[0], -1)
            k = 3
            topk_vals = np.partition(flat, -k, axis=1)[:, -k:]
            exported_pooled = topk_vals.mean(axis=1)
            exported_probs = 1 / (1 + np.exp(-exported_pooled))  # Sigmoid
        else:
            # Already pooled output
            exported_probs = 1 / (1 + np.exp(-exported_output.flatten()))
        
        # Calculate metrics
        abs_diff = np.abs(pytorch_probs - exported_probs)
        
        return {
            "pytorch_probs": pytorch_probs,
            "exported_probs": exported_probs,
            "labels": labels.numpy(),
            "abs_diff": abs_diff,
        }
    
    def run_full_validation(
        self,
        dataloader,
        threshold: float = 0.5,
        tolerance: float = 0.01,
    ) -> Dict:
        """
        Run full parity validation on a dataset.
        
        Args:
            dataloader: DataLoader for test images
            threshold: Decision threshold
            tolerance: Acceptable difference tolerance
        
        Returns:
            Dict with validation results
        """
        all_pytorch_probs = []
        all_exported_probs = []
        all_labels = []
        all_diffs = []
        
        print(f"\nRunning parity validation ({self.export_type})...")
        
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"]
            labels = batch["label"]
            
            result = self.validate_batch(images, labels)
            
            all_pytorch_probs.extend(result["pytorch_probs"].tolist())
            all_exported_probs.extend(result["exported_probs"].tolist())
            all_labels.extend(result["labels"].tolist())
            all_diffs.extend(result["abs_diff"].tolist())
        
        all_pytorch_probs = np.array(all_pytorch_probs)
        all_exported_probs = np.array(all_exported_probs)
        all_labels = np.array(all_labels)
        all_diffs = np.array(all_diffs)
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        pytorch_preds = (all_pytorch_probs > threshold).astype(int)
        exported_preds = (all_exported_probs > threshold).astype(int)
        
        pytorch_auc = roc_auc_score(all_labels, all_pytorch_probs)
        exported_auc = roc_auc_score(all_labels, all_exported_probs)
        auc_delta = abs(pytorch_auc - exported_auc)
        
        pytorch_acc = accuracy_score(all_labels, pytorch_preds)
        exported_acc = accuracy_score(all_labels, exported_preds)
        
        decision_agreement = (pytorch_preds == exported_preds).mean()
        
        correlation = np.corrcoef(all_pytorch_probs, all_exported_probs)[0, 1]
        
        # Parity checks
        checks = {
            "auc_delta_ok": auc_delta < 0.02,
            "decision_agreement_ok": decision_agreement > 0.98,
            "correlation_ok": correlation > 0.99,
            "max_diff_ok": all_diffs.max() < 0.1,
            "mean_diff_ok": all_diffs.mean() < tolerance,
        }
        
        all_passed = all(checks.values())
        
        return {
            "export_type": self.export_type,
            "num_samples": len(all_labels),
            "threshold": threshold,
            "tolerance": tolerance,
            "pytorch": {
                "auc": float(pytorch_auc),
                "accuracy": float(pytorch_acc),
                "probs_mean": float(all_pytorch_probs.mean()),
                "probs_std": float(all_pytorch_probs.std()),
            },
            "exported": {
                "auc": float(exported_auc),
                "accuracy": float(exported_acc),
                "probs_mean": float(all_exported_probs.mean()),
                "probs_std": float(all_exported_probs.std()),
            },
            "comparison": {
                "auc_delta": float(auc_delta),
                "decision_agreement": float(decision_agreement),
                "correlation": float(correlation),
                "diff_mean": float(all_diffs.mean()),
                "diff_std": float(all_diffs.std()),
                "diff_max": float(all_diffs.max()),
            },
            "checks": checks,
            "all_passed": all_passed,
        }


def print_validation_report(results: Dict):
    """Print human-readable validation report."""
    print("\n" + "=" * 80)
    print(f"EXPORT PARITY VALIDATION REPORT ({results['export_type'].upper()})")
    print("=" * 80)
    
    print(f"\n📊 Dataset: {results['num_samples']} samples")
    print(f"   Threshold: {results['threshold']}, Tolerance: {results['tolerance']}")
    
    print(f"\n📈 PyTorch Model:")
    print(f"   AUC:      {results['pytorch']['auc']:.4f}")
    print(f"   Accuracy: {results['pytorch']['accuracy']:.4f}")
    print(f"   Probs:    μ={results['pytorch']['probs_mean']:.4f}, σ={results['pytorch']['probs_std']:.4f}")
    
    print(f"\n📈 Exported Model ({results['export_type'].upper()}):")
    print(f"   AUC:      {results['exported']['auc']:.4f}")
    print(f"   Accuracy: {results['exported']['accuracy']:.4f}")
    print(f"   Probs:    μ={results['exported']['probs_mean']:.4f}, σ={results['exported']['probs_std']:.4f}")
    
    print(f"\n🔍 Comparison:")
    print(f"   AUC Delta:          {results['comparison']['auc_delta']:.6f}")
    print(f"   Decision Agreement: {results['comparison']['decision_agreement']:.2%}")
    print(f"   Correlation:        {results['comparison']['correlation']:.6f}")
    print(f"   Diff Mean:          {results['comparison']['diff_mean']:.6f}")
    print(f"   Diff Max:           {results['comparison']['diff_max']:.6f}")
    
    print(f"\n✓ Parity Checks:")
    for check, passed in results['checks'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {check}: {status}")
    
    overall = "✓ ALL PASSED" if results['all_passed'] else "✗ SOME FAILED"
    color = "" if results['all_passed'] else ""
    print(f"\n{'=' * 80}")
    print(f"OVERALL: {overall}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Validate export parity between PyTorch and TFLite/ONNX models"
    )
    parser.add_argument(
        "--pytorch-checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint",
    )
    parser.add_argument(
        "--tflite-path",
        type=str,
        default=None,
        help="Path to TFLite model (optional)",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to ONNX model (optional)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default=None,
        help="CSV split file for dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Acceptable mean difference tolerance",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for PyTorch (cuda/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/parity",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    if not args.tflite_path and not args.onnx_path:
        print("Error: Must specify at least one of --tflite-path or --onnx-path")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch model
    print("\n" + "=" * 80)
    print("LOADING PYTORCH MODEL")
    print("=" * 80)
    
    pytorch_model = LaDeDaWrapper(pretrained=False, freeze_backbone=False)
    
    checkpoint_path = Path(args.pytorch_checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    pytorch_model.model.load_state_dict(state_dict)
    pytorch_model = pytorch_model.to(args.device)
    pytorch_model.eval()
    print(f"✓ Loaded PyTorch checkpoint: {checkpoint_path}")
    
    # Load pooling
    pooling = TopKLogitPooling(r=3)
    pooling = pooling.to(args.device)
    
    # Load dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    dataset = BaseDataset(
        root_dir=args.data_root,
        split="test",
        resize_size=256,
        normalize=True,
        split_file=args.split_file,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    print(f"✓ Loaded {len(dataset)} samples")
    
    all_results = {}
    
    # Validate TFLite
    if args.tflite_path:
        print("\n" + "=" * 80)
        print("VALIDATING TFLITE MODEL")
        print("=" * 80)
        
        tflite_path = Path(args.tflite_path)
        if not tflite_path.exists():
            print(f"⚠ TFLite model not found: {tflite_path}")
        else:
            tflite_runner = TFLiteRunner(str(tflite_path))
            
            validator = ParityValidator(
                pytorch_model=pytorch_model,
                pooling=pooling,
                exported_runner=tflite_runner,
                export_type="tflite",
                device=args.device,
            )
            
            tflite_results = validator.run_full_validation(
                dataloader,
                threshold=args.threshold,
                tolerance=args.tolerance,
            )
            
            print_validation_report(tflite_results)
            all_results["tflite"] = tflite_results
    
    # Validate ONNX
    if args.onnx_path:
        print("\n" + "=" * 80)
        print("VALIDATING ONNX MODEL")
        print("=" * 80)
        
        onnx_path = Path(args.onnx_path)
        if not onnx_path.exists():
            print(f"⚠ ONNX model not found: {onnx_path}")
        else:
            onnx_runner = ONNXRunner(str(onnx_path))
            
            validator = ParityValidator(
                pytorch_model=pytorch_model,
                pooling=pooling,
                exported_runner=onnx_runner,
                export_type="onnx",
                device=args.device,
            )
            
            onnx_results = validator.run_full_validation(
                dataloader,
                threshold=args.threshold,
                tolerance=args.tolerance,
            )
            
            print_validation_report(onnx_results)
            all_results["onnx"] = onnx_results
    
    # Save results
    results_path = output_dir / "parity_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")


if __name__ == "__main__":
    main()
