#!/usr/bin/env python3
"""
Comprehensive model evaluation with all advanced metrics.

This script demonstrates how to use the full metrics suite:
- ROC-AUC, PR-AUC
- TPR@FPR=1%, FPR@TPR=95%
- Brier score, ECE
- Latency and throughput
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_comprehensive_metrics,
    compute_pr_auc,
    compute_ece,
)
from evaluation.latency import LatencyBenchmark
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.pooling import TopKLogitPooling
from datasets.base_dataset import BaseDataset


def evaluate_model_comprehensive(
    model,
    pooling,
    data_loader,
    device="cuda",
    model_name="model",
):
    """
    Run comprehensive evaluation on a model.
    
    Returns:
        dict with all metrics and statistics
    """
    print(f"\n{'=' * 80}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'=' * 80}")
    
    # =========================================================================
    # 1. Run inference and collect predictions
    # =========================================================================
    print("\n[1/3] Running inference...")
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            patch_logits = model(images)
            image_logits = pooling(patch_logits)
            probs = torch.sigmoid(image_logits.squeeze(-1))
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    print(f"✓ Inference complete: {len(all_labels)} samples")
    
    # =========================================================================
    # 2. Compute all metrics
    # =========================================================================
    print("\n[2/3] Computing metrics...")
    
    # Use comprehensive metrics function
    metrics = compute_comprehensive_metrics(all_probs, all_labels, threshold=0.5)
    
    # Compute ECE with bin details for debugging
    ece, bin_stats = compute_ece(all_labels, all_probs, n_bins=10)
    metrics["ece_bin_stats"] = bin_stats
    
    # =========================================================================
    # 3. Measure latency
    # =========================================================================
    print("\n[3/3] Measuring latency...")
    
    # Create benchmark instance
    benchmark = LatencyBenchmark(
        model,
        device=device,
        warmup_iterations=10,
        benchmark_iterations=100,
    )
    
    # Measure single image latency
    dummy_image = torch.randn(3, 256, 256).to(device)
    latency_stats = benchmark.measure_single_image_latency(dummy_image)
    
    # Measure memory
    memory_stats = benchmark.measure_memory_usage((3, 256, 256))
    
    metrics["latency"] = latency_stats
    metrics["memory"] = memory_stats
    
    # =========================================================================
    # 4. Print results
    # =========================================================================
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    
    print("\n📊 Ranking Metrics:")
    print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:   {metrics['pr_auc']:.4f}")
    
    print("\n🎯 Operating Point Metrics:")
    print(f"  TPR @ FPR=1%:     {metrics['tpr_at_1pct_fpr']:.4f} "
          f"(threshold={metrics['threshold_at_1pct_fpr']:.3f})")
    print(f"  FPR @ TPR=95%:    {metrics['fpr_at_95pct_tpr']:.4f} "
          f"(threshold={metrics['threshold_at_95pct_tpr']:.3f})")
    print(f"  Accuracy @ 0.5:   {metrics['accuracy']:.4f}")
    print(f"  Precision @ 0.5:  {metrics['precision']:.4f}")
    print(f"  Recall @ 0.5:     {metrics['recall']:.4f}")
    print(f"  F1 @ 0.5:         {metrics['f1']:.4f}")
    
    print("\n📈 Calibration Metrics:")
    print(f"  Brier Score: {metrics['brier_score']:.4f} (lower is better)")
    print(f"  ECE:         {metrics['ece']:.4f} (lower is better)")
    
    print("\n⚡ Performance Metrics:")
    print(f"  Mean Latency:   {latency_stats['mean_ms']:.2f} ms")
    print(f"  Median Latency: {latency_stats['median_ms']:.2f} ms")
    print(f"  P95 Latency:    {latency_stats['p95_ms']:.2f} ms")
    print(f"  Throughput:     {latency_stats['fps']:.1f} FPS")
    if memory_stats.get("peak_memory_mb"):
        print(f"  Peak Memory:    {memory_stats['peak_memory_mb']:.2f} MB")
    
    print("\n📋 Confusion Matrix:")
    print(f"  TP: {metrics['tp']:4d}  |  FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}  |  TN: {metrics['tn']:4d}")
    
    print(f"\n{'=' * 80}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model evaluation with advanced metrics"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/teacher/teacher_finetuned_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = LaDeDaWrapper(pretrained=False, freeze_backbone=False)
    
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.model.load_state_dict(state_dict)
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print("  Using random initialization (for testing only)")
    
    model = model.to(args.device)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = BaseDataset(
        root_dir=f"dataset/{args.split}",
        split=args.split,
        resize_size=256,
        normalize=True,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Create pooling layer
    pooling = TopKLogitPooling(r=3)
    pooling = pooling.to(args.device)
    
    # Run comprehensive evaluation
    results = evaluate_model_comprehensive(
        model,
        pooling,
        data_loader,
        device=args.device,
        model_name=checkpoint_path.stem,
    )
    
    # Save results to JSON
    output_file = output_dir / f"comprehensive_eval_{args.split}.json"
    
    # Convert numpy types to native Python for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, (np.integer, np.floating)):
            results_serializable[key] = float(value)
        elif isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value
    
    with open(output_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
