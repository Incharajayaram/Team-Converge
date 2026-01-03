#!/usr/bin/env python3
"""Evaluation script for trained student model."""

import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.student.tiny_ladeda import TinyLaDeDa
from models.pooling import TopKLogitPooling
from datasets.base_dataset import BaseDataset
from evaluation.metrics import (
    compute_roc_curve,
    compute_metrics_at_threshold,
    find_optimal_threshold
)


def load_config(config_dir="config"):
    """Load configuration files."""
    config_dir = PROJECT_ROOT / config_dir

    with open(config_dir / "base.yaml") as f:
        base_config = yaml.safe_load(f)

    with open(config_dir / "dataset.yaml") as f:
        dataset_config = yaml.safe_load(f)

    with open(config_dir / "train.yaml") as f:
        train_config = yaml.safe_load(f)

    return {**base_config, **dataset_config, **train_config}


def auto_detect_dataset_structure(dataset_root):
    """Auto-detect dataset structure and return appropriate paths."""
    dataset_root = Path(dataset_root)

    # Check if we have train/val/test subdirectories
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    # Check for test directory first
    if test_dir.exists():
        if (test_dir / "real").exists() and (test_dir / "fake").exists():
            return {
                "mode": "directory",
                "test_root": test_dir,
            }

    # Fall back to validation directory
    if val_dir.exists():
        if (val_dir / "real").exists() and (val_dir / "fake").exists():
            return {
                "mode": "directory",
                "test_root": val_dir,
            }

    # Check for CSV files
    test_csv = dataset_root / "data" / "splits" / "test.csv"
    val_csv = dataset_root / "data" / "splits" / "val.csv"

    if test_csv.exists():
        return {
            "mode": "csv",
            "test_csv": test_csv,
        }

    if val_csv.exists():
        return {
            "mode": "csv",
            "test_csv": val_csv,
        }

    return None


def create_test_loader(config, batch_size=16, num_workers=4, device="cuda"):
    """Create test data loader."""
    dataset_root = PROJECT_ROOT / config["dataset"]["root"]

    # Auto-detect dataset structure
    dataset_info = auto_detect_dataset_structure(dataset_root)

    if dataset_info is None:
        raise FileNotFoundError(
            f"Dataset structure not recognized in {dataset_root}.\n"
            "Expected either:\n"
            "1. dataset/test/{{real,fake}} or dataset/val/{{real,fake}}\n"
            "2. dataset/data/splits/test.csv or dataset/data/splits/val.csv"
        )

    print(f"\n✓ Detected dataset mode: {dataset_info['mode'].upper()}")

    if dataset_info["mode"] == "directory":
        test_dataset = BaseDataset(
            root_dir=str(dataset_info["test_root"]),
            split="test",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=None,
        )
    else:
        test_dataset = BaseDataset(
            root_dir=str(dataset_root),
            split="test",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=str(dataset_info["test_csv"]),
        )

    pin_memory = device == "cuda"

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"✓ Loaded {len(test_dataset)} test images")

    return test_loader


def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    student = TinyLaDeDa(
        pretrained=True,
        pretrained_path=str(checkpoint_path),
    )
    student = student.to(device)
    student.eval()
    print(f"✓ Loaded model from checkpoint: {checkpoint_path}")
    return student


def evaluate_model(model, test_loader, pooling=None, device="cuda"):
    """Evaluate model on test set."""
    all_logits = []
    all_labels = []
    all_probs = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Forward pass - model outputs patch-level logits
            patch_logits = model(images)

            # Pool to image-level logits
            if pooling is not None:
                image_logits = pooling(patch_logits)
            else:
                # Simple average pooling if no pooling layer
                image_logits = patch_logits.mean(dim=[2, 3], keepdim=True)

            # Convert to probabilities
            probs = torch.sigmoid(image_logits.squeeze(1))

            all_logits.append(image_logits.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    # Concatenate all batches
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()

    # Convert to binary predictions (0.5 threshold)
    all_preds = (all_probs > 0.5).astype(int)

    print(f"\n✓ Completed inference on {len(all_labels)} samples")

    return all_labels, all_probs, all_preds


def compute_metrics(labels, probs, preds):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics["confusion_matrix"] = {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }

    return metrics


def analyze_thresholds(labels, probs):
    """Analyze performance at different thresholds."""
    thresholds = np.arange(0.0, 1.01, 0.05)
    threshold_results = {}

    print("\nAnalyzing optimal thresholds...")
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(labels, probs, float(threshold))
        threshold_results[float(threshold)] = metrics

    # Find best threshold by F1 score
    best_threshold, best_f1 = find_optimal_threshold(labels, probs, metric="f1")
    print(f"✓ Best threshold (F1): {best_threshold:.2f} (F1 = {best_f1:.4f})")

    return threshold_results, best_threshold, best_f1


def save_results(results, output_path):
    """Save evaluation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")


def print_results(metrics, best_threshold=None):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print(f"\nConfusion Matrix:")
    print(f"  TP: {cm['TP']:5d}  FN: {cm['FN']:5d}")
    print(f"  FP: {cm['FP']:5d}  TN: {cm['TN']:5d}")

    if best_threshold is not None:
        print(f"\nOptimal Threshold: {best_threshold:.2f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained student model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/student_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage training checkpoint path",
    )

    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config_dir)

    # Set checkpoint path
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if args.two_stage and "checkpoints" in str(checkpoint_path):
        checkpoint_path = str(checkpoint_path).replace("checkpoints", "checkpoints_two_stage")
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  You may need to train the model first.")
        return

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {device}")

    # Create test loader
    test_loader = create_test_loader(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Load model
    model = load_model(checkpoint_path, device=device)

    # Create pooling layer
    pooling = TopKLogitPooling(r=config["pooling"]["r"])
    pooling = pooling.to(device)

    # Evaluate
    labels, probs, preds = evaluate_model(
        model, test_loader, pooling=pooling, device=device
    )

    # Load calibrated threshold if available
    calibration_json = Path(args.output_dir) / "threshold_calibration.json"
    if calibration_json.exists():
        with open(calibration_json) as f:
            calibration_data = json.load(f)
        t_star = calibration_data.get("optimal_threshold", 0.5)
        print(f"\n✓ Loaded calibrated threshold: t_star = {t_star:.2f}")
    else:
        t_star = 0.5
        print(f"\n⚠ Calibration file not found, using default threshold: 0.5")

    # Compute metrics at calibrated threshold
    preds_calibrated = (probs > t_star).astype(int)
    metrics = compute_metrics(labels, probs, preds_calibrated)

    # Analyze thresholds
    threshold_results, best_threshold_test, best_f1_test = analyze_thresholds(labels, probs)

    # Print results
    print_results(metrics, t_star)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(checkpoint_path),
        "calibrated_threshold": float(t_star),
        "metrics_at_calibrated_threshold": metrics,
        "threshold_analysis": threshold_results,
        "best_threshold_on_test": float(best_threshold_test),
        "best_f1_on_test": float(best_f1_test),
    }

    save_results(results, output_dir / "evaluation_results.json")

    # Save predictions
    predictions_data = {
        "labels": labels.tolist(),
        "probabilities": probs.tolist(),
        "predictions_at_calibrated": preds_calibrated.tolist(),
        "predictions_at_0.5": preds.tolist(),
        "predictions_at_best_test": (probs > best_threshold_test).astype(int).tolist(),
    }
    save_results(predictions_data, output_dir / "predictions.json")

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
