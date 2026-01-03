#!/usr/bin/env python3
"""Evaluate fine-tuned teacher model on validation set."""

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


def load_config(config_dir="config"):
    """Load all configuration files and merge them."""
    config_dir = Path(config_dir)

    with open(config_dir / "base.yaml") as f:
        base_config = yaml.safe_load(f)

    with open(config_dir / "dataset.yaml") as f:
        dataset_config = yaml.safe_load(f)

    with open(config_dir / "train.yaml") as f:
        train_config = yaml.safe_load(f)

    return {**base_config, **dataset_config, **train_config}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned teacher model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/teacher/teacher_finetuned_best.pth",
        help="Path to fine-tuned teacher checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    config = load_config()
    device = args.device

    print("\n" + "=" * 80)
    print("TEACHER EVALUATION: Fine-tuned Teacher on Validation Set")
    print("=" * 80)

    # =========================================================================
    # Load Models
    # =========================================================================
    print(f"\nLoading teacher from checkpoint...")

    # Create teacher model (don't load pretrained weights)
    teacher = LaDeDaWrapper(
        pretrained=False,  # Will load from checkpoint instead
        freeze_backbone=False,
    )

    # Load fine-tuned weights
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  Make sure to run: python3 scripts/train_teacher.py")
        return

    try:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        teacher.model.load_state_dict(state_dict)
        print(f"✓ Loaded fine-tuned teacher from: {checkpoint_path}")
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return

    teacher = teacher.to(device)
    teacher.eval()

    # =========================================================================
    # Load Validation Dataset
    # =========================================================================
    print(f"\nLoading validation dataset...")
    val_dataset = BaseDataset(
        root_dir="dataset/val",
        split="val",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )
    print(f"✓ Loaded {len(val_dataset)} validation samples")

    # =========================================================================
    # Setup Pooling
    # =========================================================================
    pooling = TopKLogitPooling(r=config["pooling"]["r"])
    pooling = pooling.to(device)

    # =========================================================================
    # Run Inference
    # =========================================================================
    print("\nRunning inference on validation set...")
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
            probs = torch.sigmoid(image_logits.squeeze(-1))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Collect patch stats for first batch only
            if batch_idx == 0:
                all_patch_stats.append(
                    {
                        "shape": patch_logits.shape,
                        "min": patch_logits.min().item(),
                        "max": patch_logits.max().item(),
                        "mean": patch_logits.mean().item(),
                        "std": patch_logits.std().item(),
                    }
                )

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # =========================================================================
    # Compute Metrics
    # =========================================================================
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n✓ Fine-tuned Teacher AUC: {auc:.4f}")
    print(f"✓ Fine-tuned Teacher Accuracy (threshold=0.5): {acc:.4f}")

    print(f"\n✓ Patch-logit statistics (first batch):")
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

    # =========================================================================
    # Interpretation
    # =========================================================================
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if auc < 0.55:
        print(
            f"\n⚠ WARNING: Fine-tuned Teacher AUC is {auc:.4f} (barely better than random)"
        )
        print("\nPossible causes:")
        print("  1. Dataset issue: Labels may not be reliable")
        print("  2. Insufficient training: Try more epochs or higher learning rate")
        print("  3. Architecture mismatch: Teacher may not be suitable for your data")
        print("\nRecommendation:")
        print("  → Consider skipping distillation and training student with BCE only")
        print("  → Or re-examine your dataset labels")
    elif auc < 0.65:
        print(f"\n⚠ MARGINAL: Fine-tuned Teacher AUC is {auc:.4f} (below ideal)")
        print("\nRecommendation:")
        print("  → You can proceed with distillation, but results may be limited")
        print("  → Consider starting with small alpha_distill (e.g., 0.01)")
        print("  → Train student with BCE as primary loss")
    else:
        print(f"\n✓ GOOD: Fine-tuned Teacher AUC is {auc:.4f} (informative)")
        print("\nRecommendation:")
        print("  → Fine-tuned teacher is ready for distillation")
        print("  → Proceed to student training with knowledge distillation")
        print("  → You can use standard alpha_distill values (0.1 - 0.5)")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if auc >= 0.55:
        print(f"\n✓ Fine-tuned teacher is ready!")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"\n  To train student with this teacher:")
        print(f"    1. Update config/train.yaml with teacher checkpoint path")
        print(f"    2. Run: python3 scripts/train_student_two_stage.py")
    else:
        print(f"\n✗ Consider alternative approaches:")
        print(f"  1. Train student with BCE loss only (no distillation)")
        print(f"  2. Re-examine your dataset labels")
        print(f"  3. Try different teacher architecture")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
