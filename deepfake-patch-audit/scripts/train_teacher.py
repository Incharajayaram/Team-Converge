#!/usr/bin/env python3
"""Fine-tune pretrained teacher model on your training set with early stopping."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import yaml
from tqdm import tqdm

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

    parser = argparse.ArgumentParser(
        description="Fine-tune teacher model on your training set with early stopping"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="weights/teacher", help="Output directory")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=1,
        help="Number of final blocks to unfreeze (1=layer1 only, 2=layer2+layer1, etc.)",
    )

    args = parser.parse_args()

    config = load_config()
    device = args.device

    print("\n" + "=" * 80)
    print("TEACHER FINE-TUNING WITH EARLY STOPPING")
    print("=" * 80)

    # =========================================================================
    # Load Models and Datasets
    # =========================================================================
    print("\nLoading teacher model...")
    teacher = LaDeDaWrapper(
        pretrained=True,
        pretrained_path=config["model"]["teacher"].get(
            "pretrained_path", "weights/teacher/WildRF_LaDeDa.pth"
        ),
        freeze_backbone=False,  # Don't freeze initially
    )
    teacher = teacher.to(device)
    teacher.eval()  # Start in eval mode
    print(f"✓ Loaded teacher from: {config['model']['teacher']['pretrained_path']}")

    # Get the actual model for layer access
    model = teacher.model

    # =========================================================================
    # Unfreeze last blocks for training
    # =========================================================================
    print(f"\nUnfreezing last {args.unfreeze_blocks} block(s) for fine-tuning...")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final blocks based on args.unfreeze_blocks
    trainable_params = []

    if args.unfreeze_blocks >= 1:
        # Unfreeze layer1 (last residual block)
        for param in model.layer1.parameters():
            param.requires_grad = True
        trainable_params.extend(model.layer1.parameters())
        print("  ✓ Unfrozen: layer1 (residual blocks)")

    # Note: Teacher architecture may not have layer2, layer3, etc.
    # Adjust based on actual model structure

    # Also unfreeze final classifier and normalization layers
    if hasattr(model, "bn_final"):
        for param in model.bn_final.parameters():
            param.requires_grad = True
        trainable_params.extend(model.bn_final.parameters())
        print("  ✓ Unfrozen: bn_final")

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
        trainable_params.extend(model.fc.parameters())
        print("  ✓ Unfrozen: fc")

    print(f"  Total trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    # =========================================================================
    # Load Datasets
    # =========================================================================
    print("\nLoading datasets...")
    train_dataset = BaseDataset(
        root_dir="dataset/train",
        split="train",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )
    val_dataset = BaseDataset(
        root_dir="dataset/val",
        split="val",
        resize_size=config["dataset"]["resize_size"],
        normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
    )

    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")

    # =========================================================================
    # Setup Loss and Optimizer
    # =========================================================================
    print("\nSetting up loss and optimizer...")

    # Use BCE loss for fine-tuning (teacher pooling)
    pooling = TopKLogitPooling(r=config["pooling"]["r"])
    pooling = pooling.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    print(f"✓ Loss: BCEWithLogitsLoss")
    print(f"✓ Optimizer: Adam (lr={args.lr})")
    print(f"✓ Scheduler: ReduceLROnPlateau")

    # =========================================================================
    # Training Loop with Early Stopping
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_auc = 0.0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # =====================================================================
        # Training epoch
        # =====================================================================
        teacher.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}] Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()

            # Forward pass: get patch logits
            patch_logits = teacher(images)  # (B, 1, 31, 31)

            # Pool to image level
            image_logits = pooling(patch_logits).squeeze(-1)  # (B, 1)

            # BCE loss
            loss = criterion(image_logits, labels.unsqueeze(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # =====================================================================
        # Validation epoch
        # =====================================================================
        teacher.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float()

                patch_logits = teacher(images)
                image_logits = pooling(patch_logits).squeeze(-1)

                loss = criterion(image_logits, labels.unsqueeze(-1))
                val_loss += loss.item()

                # Collect predictions
                probs = torch.sigmoid(image_logits.squeeze()).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        val_auc = roc_auc_score(all_labels, all_preds)
        val_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # =====================================================================
        # Early stopping and checkpointing
        # =====================================================================
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = output_dir / f"teacher_finetuned_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (AUC: {val_auc:.4f}) to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n✓ Early stopping triggered (patience={args.patience})")
                break

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEACHER FINE-TUNING COMPLETE")
    print("=" * 80)
    print(f"\n✓ Best validation AUC: {best_val_auc:.4f}")
    print(f"✓ Best model saved to: {output_dir / 'teacher_finetuned_best.pth'}")
    print("\nNext step: Evaluate fine-tuned teacher with diagnose_teacher.py")
    print("  python3 scripts/diagnose_teacher.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
