#!/usr/bin/env python3
"""Two-stage student model training script with progressive unfreezing."""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student.tiny_ladeda import TinyLaDeDa
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from losses.distillation import PatchDistillationLoss
from models.pooling import TopKLogitPooling
from training.train_student_two_stage import TwoStagePatchStudentTrainer


def load_config(config_path="config/base.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def auto_detect_dataset_structure(dataset_root):
    """Auto-detect dataset structure and return appropriate paths."""
    from pathlib import Path

    dataset_root = Path(dataset_root)

    # Check if we have train/val subdirectories
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    # Check if train/val have real/fake subdirectories
    if (train_dir / "real").exists() and (train_dir / "fake").exists():
        if (val_dir / "real").exists() and (val_dir / "fake").exists():
            return {
                "mode": "directory",
                "train_root": str(train_dir),
                "val_root": str(val_dir),
            }

    # Check for CSV files
    train_csv = dataset_root / "data" / "splits" / "train.csv"
    val_csv = dataset_root / "data" / "splits" / "val.csv"

    if train_csv.exists() and val_csv.exists():
        return {
            "mode": "csv",
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage training for student model with knowledge distillation"
    )
    parser.add_argument("--epochs-s1", type=int, default=5, help="Stage 1 epochs")
    parser.add_argument("--epochs-s2", type=int, default=20, help="Stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr-s1", type=float, default=0.001, help="Stage 1 learning rate")
    parser.add_argument(
        "--lr-s2", type=float, default=0.0001, help="Stage 2 learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--dataset-root", type=str, default="dataset", help="Dataset root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints_two_stage",
        help="Output directory",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        choices=["wildrf", "forensyth", "finetuned"],
        help="Teacher model weights: 'wildrf'/'forensyth' (pretrained) or 'finetuned' (fine-tuned)",
    )

    args = parser.parse_args()

    # Auto-detect dataset structure
    dataset_info = auto_detect_dataset_structure(args.dataset_root)

    if dataset_info is None:
        print("\n" + "=" * 80)
        print("ERROR: Dataset structure not recognized!")
        print("=" * 80)
        print("\nExpected one of:")
        print("1. Directory structure:")
        print("   dataset/train/real/")
        print("   dataset/train/fake/")
        print("   dataset/val/real/")
        print("   dataset/val/fake/")
        print("\n2. CSV files:")
        print("   dataset/data/splits/train.csv")
        print("   dataset/data/splits/val.csv")
        exit(1)

    print("\n" + "=" * 80)
    print("Dataset Auto-Detection")
    print("=" * 80)
    print(f"✓ Detected mode: {dataset_info['mode'].upper()}")
    if dataset_info["mode"] == "directory":
        print(f"  Train: {dataset_info['train_root']}")
        print(f"  Val:   {dataset_info['val_root']}")
    else:
        print(f"  Train CSV: {dataset_info['train_csv']}")
        print(f"  Val CSV:   {dataset_info['val_csv']}")

    # Load config
    config = load_config()

    # Override teacher weights if specified via command-line
    if args.teacher_weights:
        if args.teacher_weights.lower() == "wildrf":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/WildRF_LaDeDa.pth"
        elif args.teacher_weights.lower() == "forensyth":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/ForenSynth_LaDeDa.pth"
        elif args.teacher_weights.lower() == "finetuned":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/teacher_finetuned_best.pth"
            config["model"]["teacher"]["pretrained"] = True  # Will load checkpoint

    print("\n" + "=" * 80)
    print("TWO-STAGE STUDENT TRAINING - Configuration")
    print("=" * 80)
    print(f"Stage 1 epochs: {args.epochs_s1}")
    print(f"Stage 2 epochs: {args.epochs_s2}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stage 1 LR: {args.lr_s1}")
    print(f"Stage 2 LR: {args.lr_s2}")
    print(f"Device: {args.device}")
    print(f"Teacher weights: {config['model']['teacher']['pretrained_path']}")

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    # Student model
    student_model = TinyLaDeDa(
        pretrained=config["model"]["student"].get("pretrained", False),
        pretrained_path=config["model"]["student"].get(
            "pretrained_path", "weights/student/ForenSynth_Tiny_LaDeDa.pth"
        ),
    )
    print(f"✓ Student model loaded: {student_model.count_parameters()} parameters")

    # Teacher model
    teacher_model = LaDeDaWrapper(
        pretrained=config["model"]["teacher"].get("pretrained", True),
        pretrained_path=config["model"]["teacher"].get(
            "pretrained_path", "weights/teacher/WildRF_LaDeDa.pth"
        ),
    )
    print(f"✓ Teacher model loaded")

    # =========================================================================
    # Load Datasets
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)

    # Load datasets based on detected structure
    if dataset_info["mode"] == "directory":
        # Use directory structure (train/real, train/fake, etc.)
        train_dataset = BaseDataset(
            root_dir=dataset_info["train_root"],
            split="train",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=None,  # Use directory structure
        )

        val_dataset = BaseDataset(
            root_dir=dataset_info["val_root"],
            split="val",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=None,  # Use directory structure
        )
    else:
        # Use CSV split files
        train_dataset = BaseDataset(
            root_dir=args.dataset_root,
            split="train",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=dataset_info["train_csv"],
        )

        val_dataset = BaseDataset(
            root_dir=args.dataset_root,
            split="val",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=dataset_info["val_csv"],
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=config["dataset"]["pin_memory"],
    )

    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")

    # =========================================================================
    # Setup Loss, Pooling, and Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting Up Loss and Pooling")
    print("=" * 80)

    criterion = PatchDistillationLoss(
        alpha_distill=config["training"]["distillation"].get("alpha_distill", 0.5),
        alpha_task=config["training"]["distillation"].get("alpha_task", 0.5),
    )
    print(
        f"✓ Loss: PatchDistillationLoss "
        f"(alpha_distill={config['training']['distillation'].get('alpha_distill', 0.5)}, "
        f"alpha_task={config['training']['distillation'].get('alpha_task', 0.5)})"
    )

    pooling = TopKLogitPooling(
        r=config["pooling"].get("r", 0.1),
        min_k=config["pooling"].get("min_k", 5),
        aggregation=config["pooling"].get("aggregation", "mean"),
    )
    print(f"✓ Pooling: TopKLogitPooling (r={config['pooling'].get('r', 0.1)})")

    # =========================================================================
    # Create Trainer and Train
    # =========================================================================
    trainer = TwoStagePatchStudentTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=args.device,
        stage1_epochs=args.epochs_s1,
        stage2_epochs=args.epochs_s2,
        stage1_lr=args.lr_s1,
        stage2_lr=args.lr_s2,
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    history = trainer.train(checkpoint_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
