#!/usr/bin/env python3
"""Diagnostic: Compare teacher vs student patch logit scales."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa
from datasets.base_dataset import BaseDataset


def load_config(config_dir="config"):
    """Load all configuration files and merge them."""
    from pathlib import Path
    config_dir = Path(config_dir)

    with open(config_dir / "base.yaml") as f:
        base_config = yaml.safe_load(f)

    with open(config_dir / "dataset.yaml") as f:
        dataset_config = yaml.safe_load(f)

    with open(config_dir / "train.yaml") as f:
        train_config = yaml.safe_load(f)

    return {**base_config, **dataset_config, **train_config}


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print("LOGIT DIAGNOSTIC: Compare teacher vs student patch scales")
    print("=" * 80)

    # Load models
    teacher = LaDeDaWrapper(
        pretrained=True,
        pretrained_path=config["model"]["teacher"].get(
            "pretrained_path", "weights/teacher/WildRF_LaDeDa.pth"
        ),
    )
    teacher = teacher.to(device)
    teacher.eval()
    print(f"✓ Loaded teacher")

    student = TinyLaDeDa(
        pretrained=config["model"]["student"].get("pretrained", False),
        pretrained_path=config["model"]["student"].get(
            "pretrained_path", "weights/student/ForenSynth_Tiny_LaDeDa.pth"
        ),
    )
    student = student.to(device)
    student.eval()
    print(f"✓ Loaded student ({student.count_parameters()} parameters)")

    # Load one batch
    train_dataset = BaseDataset(
        root_dir="dataset/train",
        split="train",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print("\nGetting sample batch...")
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Forward passes
    with torch.no_grad():
        teacher_patches = teacher(images)
        student_patches = student(images)

    print("\n" + "=" * 80)
    print("PATCH LOGIT STATISTICS")
    print("=" * 80)

    print(f"\n📊 TEACHER patch logits:")
    print(f"  Shape:  {teacher_patches.shape}  (B=4, C=1, H=31, W=31)")
    print(f"  Min:    {teacher_patches.min():.6f}")
    print(f"  Max:    {teacher_patches.max():.6f}")
    print(f"  Mean:   {teacher_patches.mean():.6f}")
    print(f"  Std:    {teacher_patches.std():.6f}")

    print(f"\n📊 STUDENT patch logits (before alignment):")
    print(f"  Shape:  {student_patches.shape}  (B=4, C=1, H=126, W=126)")
    print(f"  Min:    {student_patches.min():.6f}")
    print(f"  Max:    {student_patches.max():.6f}")
    print(f"  Mean:   {student_patches.mean():.6f}")
    print(f"  Std:    {student_patches.std():.6f}")

    # Align student to teacher size
    import torch.nn.functional as F
    student_aligned = F.adaptive_avg_pool2d(student_patches, (31, 31))
    print(f"\n📊 STUDENT patch logits (after alignment to 31x31):")
    print(f"  Shape:  {student_aligned.shape}")
    print(f"  Min:    {student_aligned.min():.6f}")
    print(f"  Max:    {student_aligned.max():.6f}")
    print(f"  Mean:   {student_aligned.mean():.6f}")
    print(f"  Std:    {student_aligned.std():.6f}")

    # Compute MSE
    mse = F.mse_loss(student_aligned, teacher_patches)
    print(f"\n📊 MSE between aligned student and teacher:")
    print(f"  MSE:    {mse:.6f}")
    print(f"  √MSE:   {mse.sqrt():.6f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    teacher_range = teacher_patches.max() - teacher_patches.min()
    student_range = student_aligned.max() - student_aligned.min()

    print(f"\nRange comparison:")
    print(f"  Teacher range: {teacher_range:.6f}")
    print(f"  Student range: {student_range:.6f}")
    print(f"  Ratio: {(student_range / teacher_range):.2f}x")

    print(f"\nScale analysis:")
    if abs(teacher_patches.mean()) > 1.0:
        print(f"  ⚠ Teacher mean is large: {teacher_patches.mean():.6f}")
    if abs(student_aligned.mean()) > 1.0:
        print(f"  ⚠ Student mean is large: {student_aligned.mean():.6f}")

    if student_range / teacher_range > 3 or student_range / teacher_range < 0.33:
        print(f"\n❌ MISMATCH: Student and teacher have very different scales!")
        print(f"   This causes enormous MSE values (~{mse:.1f})")
        print(f"   Solution: Normalize logits before MSE, or reduce alpha_distill")
    else:
        print(f"\n✓ Scales are similar - MSE values are reasonable")

    print("\n" + "=" * 80)
    print("NEXT STEP")
    print("=" * 80)
    print("Run: python scripts/diagnose_teacher.py")
    print("to check if teacher itself is informative on your data")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
