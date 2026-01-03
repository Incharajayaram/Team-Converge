"""
Debugging guardrails for training robustness and reproducibility.

This module implements three critical guardrails from the implementation plan:
1. Patch-map shape contract test - Ensures student/teacher patch maps align
2. Patch-map scale logging - Tracks logit statistics to catch numerical issues
3. Pooling determinism test - Fixed audit set for reproducible debugging

These guardrails prevent silent training failures and make debugging tractable.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings


# =============================================================================
# 4.1 Patch-Map Shape Contract Test
# =============================================================================

class PatchMapShapeContract:
    """
    Enforces shape contracts between teacher and student patch maps.
    
    This guardrail catches broadcasting bugs and stride/padding mismatches
    that cause silent training failures. If shapes don't align after
    adaptive pooling, training MUST stop.
    
    Expected shapes:
    - Teacher: (B, 1, 31, 31) or similar
    - Student: (B, 1, 126, 126) or similar
    - After alignment: Student pooled to teacher's spatial size
    """
    
    def __init__(
        self,
        expected_teacher_channels: int = 1,
        expected_student_channels: int = 1,
        strict: bool = True,
    ):
        """
        Args:
            expected_teacher_channels: Expected channel dimension for teacher
            expected_student_channels: Expected channel dimension for student
            strict: If True, raise error on mismatch. If False, warn only.
        """
        self.expected_teacher_channels = expected_teacher_channels
        self.expected_student_channels = expected_student_channels
        self.strict = strict
        self._validated = False
    
    def validate(
        self,
        teacher_patches: torch.Tensor,
        student_patches: torch.Tensor,
        aligned_student_patches: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Validate shape contracts between teacher and student patch maps.
        
        Args:
            teacher_patches: Teacher patch logits (B, C, H_t, W_t)
            student_patches: Student patch logits (B, C, H_s, W_s)
            aligned_student_patches: Optional aligned student patches (B, C, H_t, W_t)
        
        Returns:
            True if all contracts pass
        
        Raises:
            AssertionError: If strict=True and contracts fail
        """
        errors = []
        
        # Contract 1: Both have 4 dimensions (B, C, H, W)
        if teacher_patches.ndim != 4:
            errors.append(f"Teacher patches must be 4D, got {teacher_patches.ndim}D")
        if student_patches.ndim != 4:
            errors.append(f"Student patches must be 4D, got {student_patches.ndim}D")
        
        # Contract 2: Batch sizes match
        if teacher_patches.shape[0] != student_patches.shape[0]:
            errors.append(
                f"Batch size mismatch: teacher={teacher_patches.shape[0]}, "
                f"student={student_patches.shape[0]}"
            )
        
        # Contract 3: Channel dimensions match expected values
        if teacher_patches.shape[1] != self.expected_teacher_channels:
            errors.append(
                f"Teacher channels mismatch: expected={self.expected_teacher_channels}, "
                f"got={teacher_patches.shape[1]}"
            )
        if student_patches.shape[1] != self.expected_student_channels:
            errors.append(
                f"Student channels mismatch: expected={self.expected_student_channels}, "
                f"got={student_patches.shape[1]}"
            )
        
        # Contract 4: If aligned patches provided, must match teacher spatial dims
        if aligned_student_patches is not None:
            if aligned_student_patches.shape != teacher_patches.shape:
                errors.append(
                    f"Aligned student shape mismatch: expected={teacher_patches.shape}, "
                    f"got={aligned_student_patches.shape}. "
                    "This indicates adaptive pooling is not working correctly."
                )
        
        # Contract 5: Spatial dimensions are reasonable (non-zero)
        if teacher_patches.shape[2] == 0 or teacher_patches.shape[3] == 0:
            errors.append(f"Teacher spatial dims are zero: {teacher_patches.shape}")
        if student_patches.shape[2] == 0 or student_patches.shape[3] == 0:
            errors.append(f"Student spatial dims are zero: {student_patches.shape}")
        
        # Handle errors
        if errors:
            error_msg = "\n".join([f"  - {e}" for e in errors])
            full_msg = (
                f"\n{'='*60}\n"
                f"PATCH MAP SHAPE CONTRACT VIOLATED\n"
                f"{'='*60}\n"
                f"Errors found:\n{error_msg}\n\n"
                f"Teacher shape: {tuple(teacher_patches.shape)}\n"
                f"Student shape: {tuple(student_patches.shape)}\n"
                f"{'='*60}\n"
                f"FIX: Check stride/padding in student model output head.\n"
                f"{'='*60}"
            )
            
            if self.strict:
                raise AssertionError(full_msg)
            else:
                warnings.warn(full_msg)
                return False
        
        self._validated = True
        return True
    
    def log_shapes(
        self,
        teacher_patches: torch.Tensor,
        student_patches: torch.Tensor,
    ) -> Dict:
        """Log shape information for debugging."""
        return {
            "teacher_shape": list(teacher_patches.shape),
            "student_shape": list(student_patches.shape),
            "teacher_spatial": [teacher_patches.shape[2], teacher_patches.shape[3]],
            "student_spatial": [student_patches.shape[2], student_patches.shape[3]],
            "batch_size": teacher_patches.shape[0],
        }


# =============================================================================
# 4.2 Patch-Map Scale Logging
# =============================================================================

class PatchMapScaleLogger:
    """
    Logs mean/std/min/max of patch logits for both teacher and student.
    
    This catches numerical issues like:
    - Distill loss >> task loss (indicates magnitude mismatch)
    - Saturated logits (all values near 0 or very large)
    - NaN/Inf values
    
    Log these once per epoch to track training health.
    """
    
    def __init__(self):
        """Initialize the scale logger."""
        self.epoch_logs = []
        self.batch_stats = []
    
    def compute_stats(self, tensor: torch.Tensor, name: str) -> Dict:
        """
        Compute statistics for a tensor.
        
        Args:
            tensor: Input tensor (any shape)
            name: Name for logging
        
        Returns:
            Dict with mean, std, min, max, has_nan, has_inf
        """
        with torch.no_grad():
            return {
                "name": name,
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "has_nan": bool(torch.isnan(tensor).any().item()),
                "has_inf": bool(torch.isinf(tensor).any().item()),
            }
    
    def log_batch(
        self,
        teacher_patches: torch.Tensor,
        student_patches: torch.Tensor,
        distill_loss: torch.Tensor,
        task_loss: torch.Tensor,
    ) -> Dict:
        """
        Log statistics for a single batch.
        
        Args:
            teacher_patches: Teacher patch logits
            student_patches: Student patch logits
            distill_loss: Current distillation loss
            task_loss: Current task loss
        
        Returns:
            Dict with all statistics
        """
        stats = {
            "teacher": self.compute_stats(teacher_patches, "teacher_patches"),
            "student": self.compute_stats(student_patches, "student_patches"),
            "distill_loss": float(distill_loss.item()),
            "task_loss": float(task_loss.item()),
            "loss_ratio": float(distill_loss.item() / (task_loss.item() + 1e-8)),
        }
        
        self.batch_stats.append(stats)
        return stats
    
    def log_epoch(self, epoch: int) -> Dict:
        """
        Aggregate batch statistics for an epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict with aggregated epoch statistics
        """
        if not self.batch_stats:
            return {}
        
        # Aggregate across batches
        epoch_stats = {
            "epoch": epoch,
            "num_batches": len(self.batch_stats),
            "teacher": {
                "mean_of_means": np.mean([b["teacher"]["mean"] for b in self.batch_stats]),
                "mean_of_stds": np.mean([b["teacher"]["std"] for b in self.batch_stats]),
                "overall_min": min([b["teacher"]["min"] for b in self.batch_stats]),
                "overall_max": max([b["teacher"]["max"] for b in self.batch_stats]),
                "any_nan": any([b["teacher"]["has_nan"] for b in self.batch_stats]),
                "any_inf": any([b["teacher"]["has_inf"] for b in self.batch_stats]),
            },
            "student": {
                "mean_of_means": np.mean([b["student"]["mean"] for b in self.batch_stats]),
                "mean_of_stds": np.mean([b["student"]["std"] for b in self.batch_stats]),
                "overall_min": min([b["student"]["min"] for b in self.batch_stats]),
                "overall_max": max([b["student"]["max"] for b in self.batch_stats]),
                "any_nan": any([b["student"]["has_nan"] for b in self.batch_stats]),
                "any_inf": any([b["student"]["has_inf"] for b in self.batch_stats]),
            },
            "loss": {
                "avg_distill_loss": np.mean([b["distill_loss"] for b in self.batch_stats]),
                "avg_task_loss": np.mean([b["task_loss"] for b in self.batch_stats]),
                "avg_loss_ratio": np.mean([b["loss_ratio"] for b in self.batch_stats]),
            },
        }
        
        self.epoch_logs.append(epoch_stats)
        self.batch_stats = []  # Reset for next epoch
        
        return epoch_stats
    
    def check_warnings(self, epoch_stats: Dict) -> List[str]:
        """
        Check for warning conditions in epoch statistics.
        
        Args:
            epoch_stats: Statistics from log_epoch()
        
        Returns:
            List of warning messages (empty if all good)
        """
        warnings_list = []
        
        # Check for NaN/Inf
        if epoch_stats["teacher"]["any_nan"]:
            warnings_list.append("⚠ NaN detected in teacher patch logits!")
        if epoch_stats["student"]["any_nan"]:
            warnings_list.append("⚠ NaN detected in student patch logits!")
        if epoch_stats["teacher"]["any_inf"]:
            warnings_list.append("⚠ Inf detected in teacher patch logits!")
        if epoch_stats["student"]["any_inf"]:
            warnings_list.append("⚠ Inf detected in student patch logits!")
        
        # Check for loss imbalance
        loss_ratio = epoch_stats["loss"]["avg_loss_ratio"]
        if loss_ratio > 100:
            warnings_list.append(
                f"⚠ Distill loss >> Task loss (ratio={loss_ratio:.1f}). "
                "Consider reducing alpha_distill or normalizing logits."
            )
        elif loss_ratio < 0.01:
            warnings_list.append(
                f"⚠ Distill loss << Task loss (ratio={loss_ratio:.4f}). "
                "Distillation may not be effective."
            )
        
        # Check for saturated logits
        teacher_range = epoch_stats["teacher"]["overall_max"] - epoch_stats["teacher"]["overall_min"]
        student_range = epoch_stats["student"]["overall_max"] - epoch_stats["student"]["overall_min"]
        
        if teacher_range < 0.01:
            warnings_list.append(
                f"⚠ Teacher logits have very narrow range ({teacher_range:.4f}). "
                "May indicate dead neurons or bad initialization."
            )
        if student_range < 0.01:
            warnings_list.append(
                f"⚠ Student logits have very narrow range ({student_range:.4f}). "
                "May indicate dead neurons or training collapse."
            )
        
        return warnings_list
    
    def print_epoch_summary(self, epoch_stats: Dict):
        """Print human-readable epoch summary."""
        print(f"\n  📊 Patch-Map Scale Stats (Epoch {epoch_stats['epoch']}):")
        print(f"     Teacher: mean={epoch_stats['teacher']['mean_of_means']:.4f}, "
              f"std={epoch_stats['teacher']['mean_of_stds']:.4f}, "
              f"range=[{epoch_stats['teacher']['overall_min']:.4f}, "
              f"{epoch_stats['teacher']['overall_max']:.4f}]")
        print(f"     Student: mean={epoch_stats['student']['mean_of_means']:.4f}, "
              f"std={epoch_stats['student']['mean_of_stds']:.4f}, "
              f"range=[{epoch_stats['student']['overall_min']:.4f}, "
              f"{epoch_stats['student']['overall_max']:.4f}]")
        print(f"     Loss ratio (distill/task): {epoch_stats['loss']['avg_loss_ratio']:.4f}")
        
        # Print warnings
        warnings_list = self.check_warnings(epoch_stats)
        for warning in warnings_list:
            print(f"     {warning}")
    
    def save_logs(self, output_path: str):
        """Save all epoch logs to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.epoch_logs, f, indent=2)


# =============================================================================
# 4.3 Pooling Determinism Test
# =============================================================================

class PoolingDeterminismTest:
    """
    Tracks pooling outputs for a fixed audit set across training.
    
    This catches:
    - Non-deterministic behavior from preprocessing or pooling
    - Tensor shape issues causing unpredictable results
    - Gradual drift in model outputs
    
    Usage:
    1. Initialize with a fixed set of images (e.g., 50 from validation)
    2. Run at the start of training to get baseline
    3. Run periodically to check for drift/issues
    """
    
    def __init__(
        self,
        audit_images: torch.Tensor,
        audit_labels: torch.Tensor,
        device: str = "cuda",
    ):
        """
        Args:
            audit_images: Fixed set of images (N, C, H, W)
            audit_labels: Labels for audit images (N,)
            device: Device for computation
        """
        self.audit_images = audit_images.to(device)
        self.audit_labels = audit_labels.to(device)
        self.device = device
        
        # History of outputs
        self.baseline = None
        self.history = []
    
    @classmethod
    def from_dataloader(
        cls,
        dataloader,
        num_samples: int = 50,
        device: str = "cuda",
        seed: int = 42,
    ) -> "PoolingDeterminismTest":
        """
        Create audit set from dataloader.
        
        Args:
            dataloader: DataLoader to sample from
            num_samples: Number of samples for audit set
            device: Device for computation
            seed: Random seed for reproducibility
        
        Returns:
            PoolingDeterminismTest instance
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        images = []
        labels = []
        count = 0
        
        for batch in dataloader:
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            for i in range(batch_images.shape[0]):
                if count >= num_samples:
                    break
                images.append(batch_images[i])
                labels.append(batch_labels[i])
                count += 1
            
            if count >= num_samples:
                break
        
        audit_images = torch.stack(images)
        audit_labels = torch.stack(labels)
        
        return cls(audit_images, audit_labels, device)
    
    def run_audit(
        self,
        model: nn.Module,
        pooling: nn.Module,
        epoch: int,
        teacher_model: Optional[nn.Module] = None,
    ) -> Dict:
        """
        Run determinism audit on fixed image set.
        
        Args:
            model: Student model
            pooling: Pooling layer
            epoch: Current epoch number
            teacher_model: Optional teacher model for comparison
        
        Returns:
            Dict with audit results
        """
        model.eval()
        
        with torch.no_grad():
            # Get student outputs
            student_patches = model(self.audit_images)
            student_pooled = pooling(student_patches)
            student_probs = torch.sigmoid(student_pooled.squeeze(-1))
            
            # Get teacher outputs if provided
            teacher_probs = None
            if teacher_model is not None:
                teacher_model.eval()
                teacher_patches = teacher_model(self.audit_images)
                teacher_pooled = pooling(teacher_patches)
                teacher_probs = torch.sigmoid(teacher_pooled.squeeze(-1))
        
        # Compute results
        results = {
            "epoch": epoch,
            "student_probs": student_probs.cpu().tolist(),
            "student_patches_shape": list(student_patches.shape),
            "pooled_logits": student_pooled.squeeze(-1).cpu().tolist(),
        }
        
        if teacher_probs is not None:
            results["teacher_probs"] = teacher_probs.cpu().tolist()
        
        # Set baseline on first run
        if self.baseline is None:
            self.baseline = {
                "student_probs": student_probs.cpu(),
                "pooled_logits": student_pooled.squeeze(-1).cpu(),
            }
            results["is_baseline"] = True
        else:
            # Compare to baseline
            prob_diff = torch.abs(
                student_probs.cpu() - self.baseline["student_probs"]
            )
            results["prob_diff_mean"] = float(prob_diff.mean().item())
            results["prob_diff_max"] = float(prob_diff.max().item())
            results["is_baseline"] = False
        
        self.history.append(results)
        return results
    
    def check_determinism(self) -> Tuple[bool, List[str]]:
        """
        Check if outputs are deterministic across runs.
        
        Returns:
            (is_deterministic, list of warning messages)
        """
        if len(self.history) < 2:
            return True, []
        
        warnings_list = []
        is_deterministic = True
        
        # Check first two runs (should be identical if deterministic)
        if len(self.history) >= 2:
            first = self.history[0]
            second = self.history[1]
            
            if first.get("is_baseline") and second.get("is_baseline") is False:
                if "prob_diff_max" in second:
                    # Allow small numerical differences
                    if second["prob_diff_max"] > 0.01:
                        is_deterministic = False
                        warnings_list.append(
                            f"⚠ Non-deterministic behavior detected! "
                            f"Max prob difference: {second['prob_diff_max']:.4f}"
                        )
        
        return is_deterministic, warnings_list
    
    def print_audit_summary(self, results: Dict):
        """Print audit results summary."""
        print(f"\n  🔍 Pooling Determinism Audit (Epoch {results['epoch']}):")
        print(f"     Audit set size: {len(results['student_probs'])}")
        print(f"     Student patch shape: {results['student_patches_shape']}")
        
        probs = results["student_probs"]
        print(f"     Prob stats: min={min(probs):.4f}, max={max(probs):.4f}, "
              f"mean={np.mean(probs):.4f}")
        
        if not results.get("is_baseline", True):
            print(f"     Diff from baseline: mean={results['prob_diff_mean']:.6f}, "
                  f"max={results['prob_diff_max']:.6f}")
    
    def save_history(self, output_path: str):
        """Save audit history to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# Integrated Guardrails Manager
# =============================================================================

class TrainingGuardrails:
    """
    Unified manager for all training guardrails.
    
    Usage:
        guardrails = TrainingGuardrails.from_config(
            val_loader=val_loader,
            output_dir="outputs/guardrails",
            device="cuda",
        )
        
        # In training loop:
        guardrails.validate_shapes(teacher_patches, student_patches, aligned)
        guardrails.log_batch_stats(teacher_patches, student_patches, distill_loss, task_loss)
        
        # At epoch end:
        guardrails.end_epoch(epoch, model, pooling)
    """
    
    def __init__(
        self,
        shape_contract: PatchMapShapeContract,
        scale_logger: PatchMapScaleLogger,
        determinism_test: Optional[PoolingDeterminismTest],
        output_dir: str = "outputs/guardrails",
    ):
        """
        Args:
            shape_contract: Shape validation instance
            scale_logger: Scale logging instance
            determinism_test: Determinism testing instance (optional)
            output_dir: Directory for saving logs
        """
        self.shape_contract = shape_contract
        self.scale_logger = scale_logger
        self.determinism_test = determinism_test
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._shape_validated_once = False
    
    @classmethod
    def from_config(
        cls,
        val_loader,
        output_dir: str = "outputs/guardrails",
        device: str = "cuda",
        expected_teacher_channels: int = 1,
        expected_student_channels: int = 1,
        audit_set_size: int = 50,
        strict_shapes: bool = True,
    ) -> "TrainingGuardrails":
        """
        Create guardrails from configuration.
        
        Args:
            val_loader: Validation DataLoader
            output_dir: Output directory for logs
            device: Device for computation
            expected_teacher_channels: Expected teacher channel dim
            expected_student_channels: Expected student channel dim
            audit_set_size: Size of determinism audit set
            strict_shapes: If True, raise on shape mismatch
        
        Returns:
            TrainingGuardrails instance
        """
        shape_contract = PatchMapShapeContract(
            expected_teacher_channels=expected_teacher_channels,
            expected_student_channels=expected_student_channels,
            strict=strict_shapes,
        )
        
        scale_logger = PatchMapScaleLogger()
        
        determinism_test = PoolingDeterminismTest.from_dataloader(
            val_loader,
            num_samples=audit_set_size,
            device=device,
        )
        
        return cls(
            shape_contract=shape_contract,
            scale_logger=scale_logger,
            determinism_test=determinism_test,
            output_dir=output_dir,
        )
    
    def validate_shapes(
        self,
        teacher_patches: torch.Tensor,
        student_patches: torch.Tensor,
        aligned_student_patches: Optional[torch.Tensor] = None,
    ):
        """Validate shape contracts (only validates once per training run)."""
        if not self._shape_validated_once:
            self.shape_contract.validate(
                teacher_patches, student_patches, aligned_student_patches
            )
            self._shape_validated_once = True
            print("  ✓ Shape contract validated")
    
    def log_batch_stats(
        self,
        teacher_patches: torch.Tensor,
        student_patches: torch.Tensor,
        distill_loss: torch.Tensor,
        task_loss: torch.Tensor,
    ):
        """Log statistics for a batch."""
        self.scale_logger.log_batch(
            teacher_patches, student_patches, distill_loss, task_loss
        )
    
    def end_epoch(
        self,
        epoch: int,
        student_model: nn.Module,
        pooling: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        print_summary: bool = True,
    ):
        """
        End-of-epoch processing: aggregate stats, run audits.
        
        Args:
            epoch: Current epoch number
            student_model: Student model
            pooling: Pooling layer
            teacher_model: Optional teacher model
            print_summary: Whether to print summary
        """
        # Aggregate scale statistics
        epoch_stats = self.scale_logger.log_epoch(epoch)
        
        if print_summary and epoch_stats:
            self.scale_logger.print_epoch_summary(epoch_stats)
        
        # Run determinism audit
        if self.determinism_test is not None:
            audit_results = self.determinism_test.run_audit(
                student_model, pooling, epoch, teacher_model
            )
            
            if print_summary:
                self.determinism_test.print_audit_summary(audit_results)
    
    def save_all_logs(self):
        """Save all guardrail logs to disk."""
        self.scale_logger.save_logs(
            str(self.output_dir / "scale_logs.json")
        )
        
        if self.determinism_test is not None:
            self.determinism_test.save_history(
                str(self.output_dir / "determinism_audit.json")
            )
        
        print(f"  ✓ Guardrail logs saved to {self.output_dir}")
