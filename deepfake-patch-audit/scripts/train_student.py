#!/usr/bin/env python3
"""Main training script for patch-level distillation training."""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa
from models.pooling import TopKLogitPooling
from losses.distillation import PatchDistillationLoss
from datasets.base_dataset import BaseDataset
from training.train_student import PatchStudentTrainer


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

    # Check if we have train/val subdirectories
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    # Check if train/val have real/fake subdirectories
    if (train_dir / "real").exists() and (train_dir / "fake").exists():
        if (val_dir / "real").exists() and (val_dir / "fake").exists():
            return {
                "mode": "directory",
                "train_root": train_dir,
                "val_root": val_dir,
            }

    # Check for CSV files
    train_csv = dataset_root / "data" / "splits" / "train.csv"
    val_csv = dataset_root / "data" / "splits" / "val.csv"

    if train_csv.exists() and val_csv.exists():
        return {
            "mode": "csv",
            "train_csv": train_csv,
            "val_csv": val_csv,
        }

    return None


def create_data_loaders(config, batch_size=16, num_workers=4, device="cuda"):
    """Create train and validation data loaders."""
    dataset_root = PROJECT_ROOT / config["dataset"]["root"]

    # Auto-detect dataset structure
    dataset_info = auto_detect_dataset_structure(dataset_root)

    if dataset_info is None:
        raise FileNotFoundError(
            f"Dataset structure not recognized in {dataset_root}.\n"
            "Expected either:\n"
            "1. dataset/train/{{real,fake}}, dataset/val/{{real,fake}}\n"
            "2. dataset/data/splits/train.csv, dataset/data/splits/val.csv"
        )

    print(f"\n✓ Detected dataset mode: {dataset_info['mode'].upper()}")

    if dataset_info["mode"] == "directory":
        # Training dataset (directory-based)
        train_dataset = BaseDataset(
            root_dir=str(dataset_info["train_root"]),
            split="train",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=None,  # Use directory structure
        )

        # Validation dataset (directory-based)
        val_dataset = BaseDataset(
            root_dir=str(dataset_info["val_root"]),
            split="val",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=None,  # Use directory structure
        )
    else:
        # Training dataset (CSV-based)
        train_dataset = BaseDataset(
            root_dir=str(dataset_root),
            split="train",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=str(dataset_info["train_csv"]),
        )

        # Validation dataset (CSV-based)
        val_dataset = BaseDataset(
            root_dir=str(dataset_root),
            split="val",
            image_format="jpg",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            normalize_mean=config["dataset"]["normalize_mean"],
            normalize_std=config["dataset"]["normalize_std"],
            split_file=str(dataset_info["val_csv"]),
        )

    # pin_memory only makes sense for CUDA training
    pin_memory = device == "cuda"

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"✓ Loaded {len(train_dataset)} training images")
    print(f"✓ Loaded {len(val_dataset)} validation images")

    return train_loader, val_loader


def create_models(config, device="cuda"):
    """Create teacher and student models."""
    # Teacher model
    teacher_path = PROJECT_ROOT / config["model"]["teacher"]["pretrained_path"]
    teacher = LaDeDaWrapper(
        pretrained=config["model"]["teacher"]["pretrained"],
        pretrained_path=str(teacher_path),
        freeze_backbone=config["model"]["teacher"]["freeze_backbone"],
    )
    teacher = teacher.to(device)
    print(f"✓ Loaded teacher model (LaDeDa9)")

    # Student model
    student_path = PROJECT_ROOT / config["model"]["student"]["pretrained_path"]
    student = TinyLaDeDa(
        pretrained=config["model"]["student"]["pretrained"],
        pretrained_path=str(student_path),
    )
    student = student.to(device)
    num_params = student.count_parameters()
    print(f"✓ Loaded student model (Tiny-LaDeDa)")
    print(f"  Student parameters: {num_params:,}")

    return teacher, student


def create_criterion_and_pooling(config, device="cuda"):
    """Create loss function and pooling."""
    # Distillation loss
    criterion = PatchDistillationLoss(
        alpha_distill=config["distillation"]["alpha_distill"],
        alpha_task=config["distillation"]["alpha_task"],
    )
    criterion = criterion.to(device)

    # Top-K pooling
    pooling = TopKLogitPooling(
        r=config["pooling"]["r"],
        min_k=config["pooling"]["min_k"],
        aggregation=config["pooling"]["aggregation"],
    )
    pooling = pooling.to(device)

    print(f"✓ Created loss function (PatchDistillationLoss)")
    print(f"✓ Created pooling (TopKLogitPooling)")

    return criterion, pooling


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Tiny-LaDeDa student with distillation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints", help="Checkpoint directory")
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        choices=["wildrf", "forensyth", "finetuned"],
        help="Teacher model weights: 'wildrf'/'forensyth' (pretrained) or 'finetuned' (fine-tuned)",
    )
    args = parser.parse_args()

    # Check CUDA availability and fall back to CPU if needed
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("\n" + "=" * 80)
    print("DEEPFAKE PATCH-AUDIT: STUDENT TRAINING")
    print("=" * 80)

    # Load config
    print("\n[1] Loading configuration...")
    config = load_config()

    # Override teacher weights if specified via command-line
    if args.teacher_weights:
        if args.teacher_weights.lower() == "wildrf":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/WildRF_LaDeDa.pth"
        elif args.teacher_weights.lower() == "forensyth":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/ForenSynth_LaDeDa.pth"
        elif args.teacher_weights.lower() == "finetuned":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/teacher_finetuned_best.pth"
            config["model"]["teacher"]["pretrained"] = True

    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        config,
        batch_size=args.batch_size,
        num_workers=4,
        device=args.device,
    )

    # Create models
    print("\n[3] Creating models...")
    teacher, student = create_models(config, device=args.device)

    # Create loss and pooling
    print("\n[4] Creating loss and pooling...")
    criterion, pooling = create_criterion_and_pooling(config, device=args.device)

    # Create trainer
    print("\n[5] Creating trainer...")
    trainer = PatchStudentTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=args.device,
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Train
    print("\n[6] Starting training...")
    history = trainer.train(
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
