"""Patch-level student model distillation training."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm


class PatchStudentTrainer:
    """
    Train student model with patch-level knowledge distillation from teacher.

    Architecture:
    - Teacher: LaDeDa9 → (B, 1, 31, 31) patch-logit map
    - Student: Tiny-LaDeDa → (B, 1, 126, 126) patch-logit map
    - Pooling: Top-K pooling on student patch logits → (B, 1) image-level prediction
    - Loss: Patch-level MSE + Image-level BCE
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        train_loader,
        val_loader,
        criterion,
        pooling,
        device="cuda",
        lr=0.001,
        weight_decay=1e-4,
    ):
        """
        Args:
            student_model: Student model to train (outputs patch-logit maps)
            teacher_model: Frozen teacher model (outputs patch-logit maps)
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: PatchDistillationLoss
            pooling: TopKLogitPooling for image-level prediction
            device: Device for training
            lr: Learning rate
            weight_decay: Weight decay
        """
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.pooling = pooling.to(device)
        self.device = device

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Optimizer for student
        self.optimizer = optim.Adam(
            self.student_model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        # Training history
        self.history = {
            "train_loss": [],
            "train_distill_loss": [],
            "train_task_loss": [],
            "val_loss": [],
            "val_distill_loss": [],
            "val_task_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

    def train_epoch(self):
        """Train for one epoch with patch-level distillation."""
        self.student_model.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Student forward pass: (B, 3, 256, 256) → (B, 1, 126, 126)
            student_patches = self.student_model(images)

            # Pool student patches for image-level prediction: (B, 1, 126, 126) → (B, 1)
            student_image_logit = self.pooling(student_patches)

            # Teacher forward pass (no grad): (B, 3, 256, 256) → (B, 1, 31, 31)
            with torch.no_grad():
                teacher_patches = self.teacher_model(images)

            # Compute patch-level distillation loss
            # Returns: (total_loss, distill_loss, task_loss)
            loss, distill_loss, task_loss = self.criterion(
                student_patches, teacher_patches, student_image_logit, labels
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        avg_task_loss = total_task_loss / len(self.train_loader)

        self.history["train_loss"].append(avg_loss)
        self.history["train_distill_loss"].append(avg_distill_loss)
        self.history["train_task_loss"].append(avg_task_loss)

        return avg_loss, avg_distill_loss, avg_task_loss

    def validate(self):
        """Validate on validation set."""
        self.student_model.eval()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Student patch-logit map: (B, 1, 126, 126)
                student_patches = self.student_model(images)

                # Pool for image-level prediction: (B, 1)
                student_image_logit = self.pooling(student_patches)

                # Teacher patch-logit map: (B, 1, 31, 31)
                teacher_patches = self.teacher_model(images)

                # Compute loss
                loss, distill_loss, task_loss = self.criterion(
                    student_patches, teacher_patches, student_image_logit, labels
                )

                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                total_task_loss += task_loss.item()

                # Compute accuracy at threshold 0.5
                predicted = (student_image_logit.squeeze(1) > 0.0).long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Store for AUC calculation
                all_preds.append(torch.sigmoid(student_image_logit.squeeze(1)).cpu())
                all_targets.append(labels.cpu())

        avg_loss = total_loss / len(self.val_loader)
        avg_distill_loss = total_distill_loss / len(self.val_loader)
        avg_task_loss = total_task_loss / len(self.val_loader)
        accuracy = correct / total

        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            auc = roc_auc_score(all_targets, all_preds)
        except:
            auc = 0.0

        self.history["val_loss"].append(avg_loss)
        self.history["val_distill_loss"].append(avg_distill_loss)
        self.history["val_task_loss"].append(avg_task_loss)
        self.history["val_acc"].append(accuracy)
        self.history["val_auc"].append(auc)

        return avg_loss, avg_distill_loss, avg_task_loss, accuracy, auc

    def train(self, epochs, checkpoint_dir="outputs/checkpoints"):
        """
        Train student model with patch-level distillation.

        Args:
            epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_auc = 0.0

        print("\n" + "=" * 80)
        print("PATCH-LEVEL DISTILLATION TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Total epochs: {epochs}")
        print(f"Checkpoint dir: {checkpoint_dir}\n")

        for epoch in range(epochs):
            # Train
            train_loss, train_distill, train_task = self.train_epoch()

            # Validate
            val_loss, val_distill, val_task, val_acc, val_auc = self.validate()

            # Step scheduler
            self.scheduler.step()

            # Print progress
            print(
                f"\nEpoch {epoch + 1}/{epochs}"
                f" | Train Loss: {train_loss:.4f}"
                f" (distill: {train_distill:.4f}, task: {train_task:.4f})"
                f" | Val Loss: {val_loss:.4f}"
                f" (distill: {val_distill:.4f}, task: {val_task:.4f})"
                f" | Acc: {val_acc:.4f}"
                f" | AUC: {val_auc:.4f}"
            )

            # Save best model based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                checkpoint_path = checkpoint_dir / "student_best.pt"
                torch.save(self.student_model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")

        # Save final model
        final_path = checkpoint_dir / "student_final.pt"
        torch.save(self.student_model.state_dict(), final_path)
        print(f"\n✓ Final model saved to {final_path}")

        # Save training history
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {history_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        return self.history
