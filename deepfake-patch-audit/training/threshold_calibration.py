"""Threshold calibration and diagnostic analysis for student model."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import csv
from typing import Tuple, Dict, Optional


class ThresholdCalibrator:
    """
    Perform threshold calibration and diagnostic analysis on validation set.

    Steps:
    1. Run inference on validation set
    2. Compute ROC-AUC
    3. Sweep thresholds to find optimal t_star
    4. Compute diagnostics at t_star
    5. Save results to JSON and optionally CSV
    """

    def __init__(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        pooling: Optional[torch.nn.Module] = None,
        device: str = "cuda",
        output_dir: str = "outputs/calibration",
    ):
        """
        Initialize calibrator.

        Args:
            model: Trained student model in eval mode
            val_loader: Validation DataLoader yielding (images, labels)
            pooling: Optional TopKLogitPooling layer
            device: Device (cuda/cpu)
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.val_loader = val_loader
        self.pooling = pooling.to(device) if pooling is not None else None
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_inference(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run model inference on validation set.

        Returns:
            y_true: Ground truth labels (shape [N], dtype int 0/1)
            p_fake: Predicted fake probabilities (shape [N], dtype float in [0,1])
        """
        all_labels = []
        all_probs = []

        print("\n" + "=" * 80)
        print("THRESHOLD CALIBRATION: Running Inference")
        print("=" * 80)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass: get patch-level logits
                patch_logits = self.model(images)

                # Pool to image-level logits
                if self.pooling is not None:
                    image_logits = self.pooling(patch_logits)
                else:
                    # Simple average pooling if no pooling layer
                    image_logits = patch_logits.mean(dim=[2, 3], keepdim=True)

                # Convert to probabilities
                p_fake = torch.sigmoid(image_logits.squeeze()).cpu().numpy()

                all_labels.append(labels.cpu().numpy())
                all_probs.append(p_fake)

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")

        y_true = np.concatenate(all_labels, axis=0).astype(int)
        p_fake = np.concatenate(all_probs, axis=0).astype(float)

        # Ensure 1D
        y_true = y_true.flatten()
        p_fake = p_fake.flatten()

        print(f"✓ Inference complete: {len(y_true)} samples")
        return y_true, p_fake

    def compute_roc_auc(self, y_true: np.ndarray, p_fake: np.ndarray) -> float:
        """Compute ROC-AUC score."""
        auc = roc_auc_score(y_true, p_fake)
        print(f"✓ ROC-AUC: {auc:.4f}")
        return auc

    def sweep_thresholds(
        self,
        y_true: np.ndarray,
        p_fake: np.ndarray,
        step: float = 0.01,
    ) -> Tuple[float, Dict]:
        """
        Sweep thresholds to find optimal t_star.

        Objective: Maximize accuracy, with ties broken by minimum FPR.

        Args:
            y_true: Ground truth labels
            p_fake: Predicted fake probabilities
            step: Threshold step (default 0.01)

        Returns:
            t_star: Optimal threshold
            results: Dict with metrics for all thresholds
        """
        print("\nThreshold sweep (optimizing for accuracy + min FPR)...")

        thresholds = np.arange(0.0, 1.0 + step, step)
        results = {}
        best_accuracy = -1.0
        best_fpr = 1.0
        t_star = 0.5

        for t in thresholds:
            y_pred = (p_fake >= t).astype(int)
            accuracy = np.mean(y_pred == y_true)

            # Compute FPR
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            results[float(t)] = {
                "accuracy": float(accuracy),
                "fpr": float(fpr),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }

            # Update best threshold
            if accuracy > best_accuracy or (
                accuracy == best_accuracy and fpr < best_fpr
            ):
                best_accuracy = accuracy
                best_fpr = fpr
                t_star = t

        print(f"✓ Optimal threshold: t_star = {t_star:.2f}")
        print(f"  Accuracy@t_star: {best_accuracy:.4f}")
        print(f"  FPR@t_star: {best_fpr:.4f}")

        return t_star, results

    def compute_diagnostics(
        self,
        y_true: np.ndarray,
        p_fake: np.ndarray,
        t_star: float,
    ) -> Dict:
        """
        Compute diagnostic statistics at optimal threshold.

        Args:
            y_true: Ground truth labels
            p_fake: Predicted fake probabilities
            t_star: Optimal threshold

        Returns:
            Diagnostics dict
        """
        y_pred = (p_fake >= t_star).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Metrics
        accuracy = np.mean(y_pred == y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Class-wise statistics
        p_fake_real = p_fake[y_true == 0]
        p_fake_fake = p_fake[y_true == 1]

        diagnostics = {
            "threshold": float(t_star),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
            },
            "confusion_matrix": {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
            },
            "class_statistics": {
                "real": {
                    "count": int(len(p_fake_real)),
                    "p_fake_mean": float(p_fake_real.mean()) if len(p_fake_real) > 0 else 0.0,
                    "p_fake_std": float(p_fake_real.std()) if len(p_fake_real) > 0 else 0.0,
                    "p_fake_min": float(p_fake_real.min()) if len(p_fake_real) > 0 else 0.0,
                    "p_fake_max": float(p_fake_real.max()) if len(p_fake_real) > 0 else 0.0,
                },
                "fake": {
                    "count": int(len(p_fake_fake)),
                    "p_fake_mean": float(p_fake_fake.mean()) if len(p_fake_fake) > 0 else 0.0,
                    "p_fake_std": float(p_fake_fake.std()) if len(p_fake_fake) > 0 else 0.0,
                    "p_fake_min": float(p_fake_fake.min()) if len(p_fake_fake) > 0 else 0.0,
                    "p_fake_max": float(p_fake_fake.max()) if len(p_fake_fake) > 0 else 0.0,
                },
            },
        }

        return diagnostics

    def compute_histogram(
        self,
        p_fake: np.ndarray,
        y_true: np.ndarray,
        bins: int = 10,
    ) -> Dict:
        """
        Compute histogram of probabilities for real vs fake.

        Args:
            p_fake: Predicted probabilities
            y_true: Ground truth labels
            bins: Number of histogram bins

        Returns:
            Histogram data
        """
        p_fake_real = p_fake[y_true == 0]
        p_fake_fake = p_fake[y_true == 1]

        edges = np.linspace(0, 1, bins + 1)
        hist_real, _ = np.histogram(p_fake_real, bins=edges)
        hist_fake, _ = np.histogram(p_fake_fake, bins=edges)

        histogram = {
            "bins": bins,
            "edges": [float(e) for e in edges],
            "real": [int(h) for h in hist_real],
            "fake": [int(h) for h in hist_fake],
        }

        return histogram

    def save_results(
        self,
        auc: float,
        y_true: np.ndarray,
        p_fake: np.ndarray,
        t_star: float,
        threshold_results: Dict,
        diagnostics: Dict,
        histogram: Dict,
        suffix: str = "",
    ) -> None:
        """
        Save calibration results to JSON and CSV.

        Args:
            auc: ROC-AUC score
            y_true: Ground truth labels
            p_fake: Predicted probabilities
            t_star: Optimal threshold
            threshold_results: Threshold sweep results
            diagnostics: Diagnostic statistics
            histogram: Histogram data
            suffix: Optional suffix for filenames (e.g., "_quantized")
        """
        # Save JSON
        results_json = {
            "auc": float(auc),
            "optimal_threshold": float(t_star),
            "diagnostics": diagnostics,
            "threshold_analysis": threshold_results,
            "histogram": histogram,
        }

        json_path = self.output_dir / f"threshold_calibration{suffix}.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"✓ Results saved to {json_path}")

        # Save predictions CSV
        csv_path = self.output_dir / f"predictions{suffix}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "y_true", "p_fake", f"y_pred@{t_star:.2f}"])
            y_pred = (p_fake >= t_star).astype(int)
            for idx, (label, prob, pred) in enumerate(zip(y_true, p_fake, y_pred)):
                writer.writerow([idx, int(label), float(prob), int(pred)])
        print(f"✓ Predictions saved to {csv_path}")

    def print_diagnostics(self, diagnostics: Dict) -> None:
        """Print diagnostics in human-readable format."""
        print("\n" + "=" * 80)
        print("THRESHOLD CALIBRATION DIAGNOSTICS")
        print("=" * 80)
        print(f"\nOptimal Threshold: {diagnostics['threshold']:.2f}")
        print("\nPerformance Metrics:")
        for key, val in diagnostics["metrics"].items():
            print(f"  {key:12s}: {val:.4f}")

        cm = diagnostics["confusion_matrix"]
        print("\nConfusion Matrix:")
        print(f"  TP: {cm['TP']:6d}  FN: {cm['FN']:6d}")
        print(f"  FP: {cm['FP']:6d}  TN: {cm['TN']:6d}")

        print("\nProbability Statistics:")
        print("  Real class (y_true=0):")
        real_stats = diagnostics["class_statistics"]["real"]
        print(f"    Count: {real_stats['count']}")
        print(f"    Mean p_fake: {real_stats['p_fake_mean']:.4f} ± {real_stats['p_fake_std']:.4f}")
        print(f"    Range: [{real_stats['p_fake_min']:.4f}, {real_stats['p_fake_max']:.4f}]")

        print("  Fake class (y_true=1):")
        fake_stats = diagnostics["class_statistics"]["fake"]
        print(f"    Count: {fake_stats['count']}")
        print(f"    Mean p_fake: {fake_stats['p_fake_mean']:.4f} ± {fake_stats['p_fake_std']:.4f}")
        print(f"    Range: [{fake_stats['p_fake_min']:.4f}, {fake_stats['p_fake_max']:.4f}]")

        print("\n" + "=" * 80)

    def calibrate(self, save_csv: bool = True, suffix: str = "") -> Tuple[float, Dict]:
        """
        Run full calibration pipeline.

        Args:
            save_csv: Whether to save predictions CSV
            suffix: Optional suffix for output filenames

        Returns:
            t_star: Optimal threshold
            results: Full results dict
        """
        # Step 1: Inference
        y_true, p_fake = self.run_inference()

        # Step 2: Compute AUC
        auc = self.compute_roc_auc(y_true, p_fake)

        # Step 3: Threshold sweep
        t_star, threshold_results = self.sweep_thresholds(y_true, p_fake)

        # Step 4: Diagnostics
        diagnostics = self.compute_diagnostics(y_true, p_fake, t_star)

        # Step 5: Histogram
        histogram = self.compute_histogram(p_fake, y_true)

        # Step 6: Save results
        self.save_results(
            auc,
            y_true,
            p_fake,
            t_star,
            threshold_results,
            diagnostics,
            histogram,
            suffix=suffix,
        )

        # Print diagnostics
        self.print_diagnostics(diagnostics)

        results = {
            "auc": auc,
            "t_star": float(t_star),
            "diagnostics": diagnostics,
            "threshold_results": threshold_results,
            "histogram": histogram,
        }

        return float(t_star), results
