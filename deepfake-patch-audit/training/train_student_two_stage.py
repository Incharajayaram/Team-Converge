"""Two-stage student model distillation training with progressive unfreezing."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm

from training.threshold_calibration import ThresholdCalibrator
from training.guardrails import TrainingGuardrails


class TwoStagePatchStudentTrainer:
    """
    Two-stage training for student model with patch-level knowledge distillation.

    Stage 1: Classifier Training
    - Freeze entire backbone (conv1, conv2, layer1)
    - Train only final classifier layer (fc)
    - High learning rate, few epochs
    - Purpose: Quick initialization of classifier

    Stage 2: Fine-tuning
    - Unfreeze last residual blocks (layer1)
    - Keep earlier layers frozen (conv1, conv2)
    - Fine-tune with smaller learning rate
    - Purpose: Adapt deeper features while preserving pretrained knowledge

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
        stage1_epochs=5,
        stage2_epochs=20,
        stage1_lr=0.001,
        stage2_lr=0.0001,
        weight_decay=1e-4,
        enable_guardrails=True,
        guardrails_output_dir="outputs/guardrails",
    ):
        """
        Args:
            student_model: Student model to train
            teacher_model: Frozen teacher model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: PatchDistillationLoss
            pooling: TopKLogitPooling for image-level prediction
            device: Device for training
            stage1_epochs: Number of epochs for stage 1 (classifier training)
            stage2_epochs: Number of epochs for stage 2 (fine-tuning)
            stage1_lr: Learning rate for stage 1
            stage2_lr: Learning rate for stage 2 (usually smaller)
            weight_decay: Weight decay for optimizer
            enable_guardrails: Enable debugging guardrails (shape/scale/determinism checks)
            guardrails_output_dir: Output directory for guardrail logs
        """
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.pooling = pooling.to(device)
        self.device = device

        # Stage configuration
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.weight_decay = weight_decay

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Training history
        self.history = {
            "stage": [],  # Track which stage each epoch belongs to
            "train_loss": [],
            "train_distill_loss": [],
            "train_task_loss": [],
            "val_loss": [],
            "val_distill_loss": [],
            "val_task_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

        # Optimizers (will be set up per stage)
        self.optimizer = None
        self.scheduler = None
        
        # Setup debugging guardrails (Item 4 from implementation plan)
        self.enable_guardrails = enable_guardrails
        self.guardrails = None
        if enable_guardrails:
            print("\n  Initializing debugging guardrails...")
            self.guardrails = TrainingGuardrails.from_config(
                val_loader=val_loader,
                output_dir=guardrails_output_dir,
                device=device,
                expected_teacher_channels=1,
                expected_student_channels=1,
                audit_set_size=50,
                strict_shapes=True,
            )
            print("  ✓ Guardrails initialized (shape contracts, scale logging, determinism tests)")

    def _freeze_backbone(self):
        """Freeze all backbone layers (everything except classifier)."""
        # Get the actual model (handle both direct model and wrapper)
        model = self.student_model.model if hasattr(self.student_model, 'model') else self.student_model

        # Freeze preprocessing and initial convolutions
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.conv2.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False

        # Freeze layer1 (residual blocks)
        for param in model.layer1.parameters():
            param.requires_grad = False

        print("✓ Frozen: conv1, conv2, bn1, layer1 (backbone)")

    def _unfreeze_layer1(self):
        """Unfreeze layer1 (residual blocks) for fine-tuning."""
        # Get the actual model (handle both direct model and wrapper)
        model = self.student_model.model if hasattr(self.student_model, 'model') else self.student_model

        for param in model.layer1.parameters():
            param.requires_grad = True

        print("✓ Unfrozen: layer1 (residual blocks)")

    def _get_trainable_params(self):
        """Return only trainable parameters."""
        return [p for p in self.student_model.parameters() if p.requires_grad]

    def _setup_stage1_optimizer(self):
        """Set up optimizer for stage 1 (classifier only)."""
        # Get the actual model (handle both direct model and wrapper)
        model = self.student_model.model if hasattr(self.student_model, 'model') else self.student_model

        # Only train classifier layer
        self.optimizer = optim.Adam(
            model.fc.parameters(), lr=self.stage1_lr, weight_decay=self.weight_decay
        )

        # Scheduler: reduce LR if no improvement
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        print(f"✓ Stage 1 Optimizer: Adam (lr={self.stage1_lr})")

    def _setup_stage2_optimizer(self):
        """Set up optimizer for stage 2 (layer1 + classifier)."""
        # Get the actual model (handle both direct model and wrapper)
        model = self.student_model.model if hasattr(self.student_model, 'model') else self.student_model

        # Train layer1 and classifier with smaller learning rate
        trainable_params = [
            {"params": model.layer1.parameters(), "lr": self.stage2_lr},
            {"params": model.fc.parameters(), "lr": self.stage2_lr},
        ]

        self.optimizer = optim.Adam(
            trainable_params, weight_decay=self.weight_decay
        )

        # Scheduler: cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.stage2_epochs, eta_min=1e-6
        )

        print(f"✓ Stage 2 Optimizer: Adam (lr={self.stage2_lr})")

    def train_epoch(self, stage=1):
        """Train for one epoch."""
        self.student_model.train()
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"[Stage {stage}] Training", leave=False)
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
            loss, distill_loss, task_loss = self.criterion(
                student_patches, teacher_patches, student_image_logit, labels
            )
            
            # Guardrails: Shape validation (first batch only) and scale logging
            if self.guardrails is not None:
                # Validate shape contracts on first batch of training
                self.guardrails.validate_shapes(
                    teacher_patches, student_patches
                )
                # Log batch statistics for scale monitoring
                self.guardrails.log_batch_stats(
                    teacher_patches, student_patches, distill_loss, task_loss
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._get_trainable_params(), max_norm=1.0
            )
            self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_distill_loss += distill_loss.item()
            total_task_loss += task_loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        avg_distill_loss = total_distill_loss / len(self.train_loader)
        avg_task_loss = total_task_loss / len(self.train_loader)

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

                student_patches = self.student_model(images)
                student_image_logit = self.pooling(student_patches)

                with torch.no_grad():
                    teacher_patches = self.teacher_model(images)

                loss, distill_loss, task_loss = self.criterion(
                    student_patches, teacher_patches, student_image_logit, labels
                )

                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                total_task_loss += task_loss.item()

                predicted = (student_image_logit.squeeze(1) > 0.0).long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

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

        return avg_loss, avg_distill_loss, avg_task_loss, accuracy, auc

    def train(self, checkpoint_dir="outputs/checkpoints_two_stage"):
        """
        Train student model with two-stage approach.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_auc = 0.0

        print("\n" + "=" * 80)
        print("TWO-STAGE PATCH-LEVEL DISTILLATION TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Stage 1: {self.stage1_epochs} epochs (classifier only, lr={self.stage1_lr})")
        print(f"Stage 2: {self.stage2_epochs} epochs (fine-tune, lr={self.stage2_lr})")
        print(f"Checkpoint dir: {checkpoint_dir}\n")

        # =====================================================================
        # STAGE 1: Train classifier layer only
        # =====================================================================
        print("=" * 80)
        print("STAGE 1: CLASSIFIER TRAINING (Backbone Frozen)")
        print("=" * 80)

        self._freeze_backbone()
        self._setup_stage1_optimizer()

        for epoch in range(self.stage1_epochs):
            train_loss, train_distill, train_task = self.train_epoch(stage=1)
            val_loss, val_distill, val_task, val_acc, val_auc = self.validate()

            # Update scheduler
            self.scheduler.step(val_auc)

            self.history["stage"].append(1)
            self.history["train_loss"].append(train_loss)
            self.history["train_distill_loss"].append(train_distill)
            self.history["train_task_loss"].append(train_task)
            self.history["val_loss"].append(val_loss)
            self.history["val_distill_loss"].append(val_distill)
            self.history["val_task_loss"].append(val_task)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            print(
                f"\nStage 1 - Epoch {epoch + 1}/{self.stage1_epochs}"
                f" | Train Loss: {train_loss:.4f}"
                f" (distill: {train_distill:.4f}, task: {train_task:.4f})"
                f" | Val Loss: {val_loss:.4f}"
                f" (distill: {val_distill:.4f}, task: {val_task:.4f})"
                f" | Acc: {val_acc:.4f}"
                f" | AUC: {val_auc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                checkpoint_path = checkpoint_dir / "student_stage1_best.pt"
                torch.save(self.student_model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best stage 1 model (AUC: {val_auc:.4f})")
            
            # Guardrails: End-of-epoch processing (scale summary + determinism audit)
            if self.guardrails is not None:
                self.guardrails.end_epoch(
                    epoch=epoch + 1,
                    student_model=self.student_model,
                    pooling=self.pooling,
                    teacher_model=self.teacher_model,
                    print_summary=True,
                )

        # =====================================================================
        # STAGE 2: Fine-tune layer1 + classifier with smaller learning rate
        # =====================================================================
        print("\n" + "=" * 80)
        print("STAGE 2: FINE-TUNING (Layer1 + Classifier Unfrozen)")
        print("=" * 80)

        self._unfreeze_layer1()
        self._setup_stage2_optimizer()

        for epoch in range(self.stage2_epochs):
            train_loss, train_distill, train_task = self.train_epoch(stage=2)
            val_loss, val_distill, val_task, val_acc, val_auc = self.validate()

            # Update scheduler
            self.scheduler.step()

            self.history["stage"].append(2)
            self.history["train_loss"].append(train_loss)
            self.history["train_distill_loss"].append(train_distill)
            self.history["train_task_loss"].append(train_task)
            self.history["val_loss"].append(val_loss)
            self.history["val_distill_loss"].append(val_distill)
            self.history["val_task_loss"].append(val_task)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            print(
                f"\nStage 2 - Epoch {epoch + 1}/{self.stage2_epochs}"
                f" | Train Loss: {train_loss:.4f}"
                f" (distill: {train_distill:.4f}, task: {train_task:.4f})"
                f" | Val Loss: {val_loss:.4f}"
                f" (distill: {val_distill:.4f}, task: {val_task:.4f})"
                f" | Acc: {val_acc:.4f}"
                f" | AUC: {val_auc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                checkpoint_path = checkpoint_dir / "student_stage2_best.pt"
                torch.save(self.student_model.state_dict(), checkpoint_path)
                print(f"  ✓ Saved best stage 2 model (AUC: {val_auc:.4f})")
            
            # Guardrails: End-of-epoch processing (scale summary + determinism audit)
            if self.guardrails is not None:
                self.guardrails.end_epoch(
                    epoch=self.stage1_epochs + epoch + 1,
                    student_model=self.student_model,
                    pooling=self.pooling,
                    teacher_model=self.teacher_model,
                    print_summary=True,
                )

        # Save final model and history
        final_path = checkpoint_dir / "student_final.pt"
        torch.save(self.student_model.state_dict(), final_path)
        print(f"\n✓ Final model saved to {final_path}")
        
        # Save guardrail logs
        if self.guardrails is not None:
            self.guardrails.save_all_logs()

        history_path = checkpoint_dir / "training_history_two_stage.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {history_path}")

        # =====================================================================
        # THRESHOLD CALIBRATION
        # =====================================================================
        print("\n" + "=" * 80)
        print("POST-TRAINING THRESHOLD CALIBRATION")
        print("=" * 80)

        calibrator = ThresholdCalibrator(
            model=self.student_model,
            val_loader=self.val_loader,
            pooling=self.pooling,
            device=self.device,
            output_dir=str(checkpoint_dir / "calibration"),
        )

        t_star, calibration_results = calibrator.calibrate(suffix="")

        # Save calibration results with training history
        self.history["threshold_calibration"] = {
            "optimal_threshold": float(t_star),
            "auc": float(calibration_results["auc"]),
            "diagnostics": calibration_results["diagnostics"],
        }

        # Update history file with calibration
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Updated training history with calibration results")

        print("\n" + "=" * 80)
        print("TWO-STAGE TRAINING COMPLETE")
        print("=" * 80)

        return self.history
