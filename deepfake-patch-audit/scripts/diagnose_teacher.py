#!/usr/bin/env python3
"""Diagnostic: Check if teacher is informative on your validation set."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.pooling import TopKLogitPooling
from datasets.base_dataset import BaseDataset


def load_config(config_path="config/base.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print("TEACHER DIAGNOSTIC: Is teacher informative on your validation set?")
    print("=" * 80)

    # Load teacher
    teacher = LaDeDaWrapper(
        pretrained=True,
        pretrained_path=config["model"]["teacher"].get(
            "pretrained_path", "weights/teacher/WildRF_LaDeDa.pth"
        ),
    )
    teacher = teacher.to(device)
    teacher.eval()
    print(f"✓ Loaded teacher from: {config['model']['teacher']['pretrained_path']}")

    # Load validation dataset
    val_dataset = BaseDataset(
        root_dir="dataset/val",
        split="val",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    print(f"✓ Loaded {len(val_dataset)} validation samples")

    # Setup pooling
    pooling = TopKLogitPooling(r=config["pooling"]["r"])
    pooling = pooling.to(device)

    # Run inference
    print("\nRunning teacher inference on validation set...")
    all_probs = []
    all_labels = []
    all_patch_stats = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Teacher forward
            patch_logits = teacher(images)
            image_logits = pooling(patch_logits)
            probs = torch.sigmoid(image_logits.squeeze())

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Collect patch stats for first batch only
            if batch_idx == 0:
                all_patch_stats.append({
                    "shape": patch_logits.shape,
                    "min": patch_logits.min().item(),
                    "max": patch_logits.max().item(),
                    "mean": patch_logits.mean().item(),
                    "std": patch_logits.std().item(),
                })

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n✓ Teacher AUC on validation set: {auc:.4f}")
    print(f"✓ Teacher Accuracy (threshold=0.5): {acc:.4f}")

    print(f"\n✓ Teacher patch-logit statistics (first batch):")
    stats = all_patch_stats[0]
    print(f"  Shape: {stats['shape']}")
    print(f"  Min:   {stats['min']:.6f}")
    print(f"  Max:   {stats['max']:.6f}")
    print(f"  Mean:  {stats['mean']:.6f}")
    print(f"  Std:   {stats['std']:.6f}")

    print(f"\n✓ Probability statistics:")
    print(f"  Min:   {all_probs.min():.6f}")
    print(f"  Max:   {all_probs.max():.6f}")
    print(f"  Mean:  {all_probs.mean():.6f}")
    print(f"  Std:   {all_probs.std():.6f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if auc < 0.55:
        print(f"\n⚠ WARNING: Teacher AUC is {auc:.4f} (barely better than random 0.50)")
        print("\nPossible causes:")
        print("  1. Domain shift: Teacher trained on different data than your dataset")
        print("  2. Dataset issue: Your real/fake labels may not match teacher's training data")
        print("  3. Preprocessing mismatch: Different image preprocessing than original")
        print("\nRecommendation:")
        print("  → Reduce alpha_distill (set to 0.01 or 0.0) and train with BCE only")
        print("  → Verify your dataset labels are correct")
        print("  → Consider fine-tuning teacher on your dataset first")
    else:
        print(f"\n✓ Good: Teacher AUC is {auc:.4f} (informative)")
        print("  Teacher should provide useful distillation signal")
        print("  Issue is likely loss scaling or alpha weights")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
