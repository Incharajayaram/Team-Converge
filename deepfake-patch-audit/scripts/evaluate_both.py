#!/usr/bin/env python3
"""Comprehensive evaluation of both teacher and student models."""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa
from models.pooling import TopKLogitPooling
from datasets.base_dataset import BaseDataset


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


def load_test_dataset(config, batch_size=32, num_workers=4, device="cuda"):
    """Load test/validation dataset."""
    test_dataset = BaseDataset(
        root_dir="dataset/val",  # Using val as test set
        split="val",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"✓ Loaded {len(test_dataset)} test samples")
    return test_loader


def evaluate_teacher(teacher, test_loader, pooling, device="cuda"):
    """Evaluate teacher model."""
    print("\n" + "=" * 80)
    print("EVALUATING TEACHER")
    print("=" * 80)

    teacher.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            patch_logits = teacher(images)
            image_logits = pooling(patch_logits)
            probs = torch.sigmoid(image_logits.squeeze(-1))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics, all_probs, all_labels


def evaluate_student(student, test_loader, pooling, device="cuda"):
    """Evaluate student model."""
    print("\n" + "=" * 80)
    print("EVALUATING STUDENT")
    print("=" * 80)

    student.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            patch_logits = student(images)
            image_logits = pooling(patch_logits)
            probs = torch.sigmoid(image_logits.squeeze(-1))

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics, all_probs, all_labels


def print_comparison(teacher_metrics, student_metrics):
    """Print side-by-side comparison of teacher vs student."""
    print("\n" + "=" * 80)
    print("TEACHER vs STUDENT COMPARISON")
    print("=" * 80)

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]

    print(f"\n{'Metric':<12} {'Teacher':<12} {'Student':<12} {'Improvement':<12}")
    print("-" * 50)

    for name, key in zip(metrics_names, metric_keys):
        teacher_val = teacher_metrics[key]
        student_val = student_metrics[key]
        improvement = student_val - teacher_val

        improvement_str = f"+{improvement:.4f}" if improvement >= 0 else f"{improvement:.4f}"
        improvement_color = "✓" if improvement >= 0 else "✗"

        print(
            f"{name:<12} {teacher_val:<12.4f} {student_val:<12.4f} {improvement_color} {improvement_str:<10}"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    avg_improvement = np.mean(
        [student_metrics[k] - teacher_metrics[k] for k in metric_keys]
    )

    if avg_improvement > 0.02:
        print(f"\n✓ EXCELLENT: Student improved by avg {avg_improvement:.4f}")
        print("  → Distillation was very effective!")
    elif avg_improvement > 0:
        print(f"\n✓ GOOD: Student improved by avg {avg_improvement:.4f}")
        print("  → Distillation helped, but improvements are modest")
    else:
        print(f"\n⚠ POOR: Student regressed by avg {abs(avg_improvement):.4f}")
        print("  → Distillation may not have helped (possible teacher quality issue)")
        print("  → Consider training student with BCE only next time")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate both teacher and student models"
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=str,
        default="weights/teacher/teacher_finetuned_best.pth",
        help="Path to fine-tuned teacher checkpoint",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=str,
        default="outputs/checkpoints_two_stage/student_final.pt",
        help="Path to trained student checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = args.device

    print("\n" + "=" * 80)
    print("TEACHER vs STUDENT EVALUATION")
    print("=" * 80)

    # Load config
    config = load_config()

    # Load test dataset
    print("\nLoading test dataset...")
    test_loader = load_test_dataset(
        config, batch_size=args.batch_size, num_workers=args.num_workers, device=device
    )

    # Setup pooling
    pooling = TopKLogitPooling(r=config["pooling"]["r"])
    pooling = pooling.to(device)

    # Load and evaluate teacher
    print("\nLoading teacher model...")
    teacher_path = Path(args.teacher_checkpoint)
    if teacher_path.exists():
        teacher = LaDeDaWrapper(pretrained=False)
        state_dict = torch.load(str(teacher_path), map_location="cpu")
        teacher.model.load_state_dict(state_dict)
        teacher = teacher.to(device)
        teacher_metrics, teacher_probs, teacher_labels = evaluate_teacher(
            teacher, test_loader, pooling, device
        )
        print(f"✓ Teacher AUC: {teacher_metrics['auc']:.4f}")
    else:
        print(f"✗ Teacher checkpoint not found: {teacher_path}")
        return

    # Load and evaluate student
    print("\nLoading student model...")
    student_path = Path(args.student_checkpoint)
    if student_path.exists():
        student = TinyLaDeDa(pretrained=True, pretrained_path=str(student_path))
        student = student.to(device)
        student_metrics, student_probs, student_labels = evaluate_student(
            student, test_loader, pooling, device
        )
        print(f"✓ Student AUC: {student_metrics['auc']:.4f}")
    else:
        print(f"✗ Student checkpoint not found: {student_path}")
        return

    # Print comparison
    print_comparison(teacher_metrics, student_metrics)

    # Print detailed metrics
    print("\n" + "=" * 80)
    print("DETAILED METRICS")
    print("=" * 80)

    print("\nTEACHER:")
    for key, value in teacher_metrics.items():
        print(f"  {key.capitalize():<12}: {value:.4f}")

    print("\nSTUDENT:")
    for key, value in student_metrics.items():
        print(f"  {key.capitalize():<12}: {value:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
