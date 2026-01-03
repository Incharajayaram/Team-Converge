#!/usr/bin/env python3
"""
Benchmark evaluation on Celeb-DF v2 dataset (TEST-ONLY).

This script evaluates a trained model on the Celeb-DF v2 benchmark
dataset without any training or threshold tuning.

CRITICAL: This is TEST-ONLY evaluation. Do NOT use Celeb-DF for:
- Training
- Validation
- Threshold calibration
- Hyperparameter tuning

Usage:
    python scripts/evaluate_benchmark.py \
        --checkpoint weights/teacher/teacher_finetuned_best.pth \
        --benchmark-root /path/to/CelebDF \
        --threshold 0.5 \
        --device cuda
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

from datasets.celebdf_dataset import CelebDFDataset
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.pooling import TopKLogitPooling
from evaluation.metrics import compute_comprehensive_metrics


def evaluate_on_benchmark(
    model,
    pooling,
    benchmark_loader,
    device="cuda",
    threshold=0.5,
    model_name="model",
):
    """
    Evaluate model on Celeb-DF benchmark.
    
    Args:
        model: Trained model
        pooling: Pooling layer
        benchmark_loader: Celeb-DF DataLoader
        device: Device to run on
        threshold: Decision threshold (from validation calibration)
        model_name: Model name for logging
    
    Returns:
        dict with benchmark results
    """
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK EVALUATION: {model_name} on Celeb-DF v2")
    print(f"{'=' * 80}")
    print(f"\n⚠ TEST-ONLY EVALUATION: No training/tuning on this data")
    print(f"  Using threshold from validation: {threshold:.3f}")
    
    # Run inference
    print(f"\nRunning inference on Celeb-DF v2...")
    model.eval()
    all_probs = []
    all_labels = []
    all_videos = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(benchmark_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            videos = batch["video"]
            
            # Forward pass
            patch_logits = model(images)
            image_logits = pooling(patch_logits)
            probs = torch.sigmoid(image_logits.squeeze(-1))
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_videos.extend(videos)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    print(f"✓ Inference complete: {len(all_labels)} frames")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_comprehensive_metrics(all_labels, all_probs, threshold=threshold)
    
    # Compute video-level metrics
    print(f"\nComputing video-level metrics...")
    video_metrics = compute_video_level_metrics(all_videos, all_probs, all_labels)
    
    # Print results
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS (Celeb-DF v2)")
    print(f"{'=' * 80}")
    
    print(f"\n📊 Frame-Level Metrics:")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:           {metrics['pr_auc']:.4f}")
    print(f"  TPR @ FPR=1%:     {metrics['tpr_at_1pct_fpr']:.4f}")
    print(f"  Accuracy @ {threshold:.2f}:  {metrics['accuracy']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"  Brier Score:      {metrics['brier_score']:.4f}")
    print(f"  ECE:              {metrics['ece']:.4f}")
    
    print(f"\n📋 Frame-Level Confusion Matrix:")
    print(f"  TP: {metrics['tp']:5d}  |  FP: {metrics['fp']:5d}")
    print(f"  FN: {metrics['fn']:5d}  |  TN: {metrics['tn']:5d}")
    
    print(f"\n🎥 Video-Level Metrics:")
    print(f"  Videos evaluated:  {video_metrics['num_videos']}")
    print(f"  Video accuracy:    {video_metrics['video_accuracy']:.4f}")
    print(f"  Video AUC:         {video_metrics['video_auc']:.4f}")
    
    print(f"\n{'=' * 80}")
    
    # Combine results
    results = {
        "frame_metrics": metrics,
        "video_metrics": video_metrics,
        "threshold": threshold,
        "benchmark_name": "Celeb-DF v2",
    }
    
    return results


def compute_video_level_metrics(video_names, frame_probs, frame_labels):
    """
    Compute video-level metrics by aggregating frame predictions.
    
    For each video, use mean probability across all frames.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Group by video
    video_data = {}
    for video, prob, label in zip(video_names, frame_probs, frame_labels):
        if video not in video_data:
            video_data[video] = {
                "probs": [],
                "label": label,  # All frames from same video have same label
            }
        video_data[video]["probs"].append(prob)
    
    # Aggregate to video level (mean probability)
    video_probs = []
    video_labels = []
    
    for video, data in video_data.items():
        video_probs.append(np.mean(data["probs"]))
        video_labels.append(data["label"])
    
    video_probs = np.array(video_probs)
    video_labels = np.array(video_labels)
    
    # Compute metrics
    video_auc = roc_auc_score(video_labels, video_probs)
    video_preds = (video_probs > 0.5).astype(int)
    video_acc = accuracy_score(video_labels, video_preds)
    
    return {
        "num_videos": len(video_data),
        "video_accuracy": video_acc,
        "video_auc": video_auc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation on Celeb-DF v2 (TEST-ONLY)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        required=True,
        help="Root directory of Celeb-DF v2 dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold (from validation calibration)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for extraction",
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
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of frames even if cache exists",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = LaDeDaWrapper(pretrained=False, freeze_backbone=False)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    
    # Load Celeb-DF dataset
    print(f"\nLoading Celeb-DF v2 benchmark...")
    benchmark_dataset = CelebDFDataset(
        root_dir=args.benchmark_root,
        fps=args.fps,
        resize_size=256,
        normalize=True,
        force_extract=args.force_extract,
    )
    
    benchmark_loader = DataLoader(
        benchmark_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    # Create pooling layer
    pooling = TopKLogitPooling(r=3)
    pooling = pooling.to(args.device)
    
    # Run benchmark evaluation
    results = evaluate_on_benchmark(
        model,
        pooling,
        benchmark_loader,
        device=args.device,
        threshold=args.threshold,
        model_name=checkpoint_path.stem,
    )
    
    # Save results
    output_file = output_dir / "celebdf_benchmark_results.json"
    
    # Convert numpy types for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}")
    
    frame_auc = results["frame_metrics"]["roc_auc"]
    video_auc = results["video_metrics"]["video_auc"]
    
    if frame_auc < 0.60:
        print(f"\n⚠ POOR: Frame AUC = {frame_auc:.4f}")
        print("  Significant domain shift from internal data to Celeb-DF.")
        print("  This is common and acceptable for a hackathon demo if:")
        print("    1. Internal dataset performance is good")
        print("    2. You acknowledge the limitation honestly")
        print("    3. You show heatmaps and explain the domain gap")
    elif frame_auc < 0.70:
        print(f"\n🟡 MODERATE: Frame AUC = {frame_auc:.4f}")
        print("  Model generalizes moderately well to Celeb-DF.")
        print("  This is acceptable for deployment with caveats.")
    else:
        print(f"\n✅ GOOD: Frame AUC = {frame_auc:.4f}")
        print("  Model generalizes well to external benchmark!")
        print("  Strong evidence of real-world viability.")
    
    print(f"\n✓ Video-level AUC = {video_auc:.4f}")
    print("  (Video-level is the primary Celeb-DF metric)")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
