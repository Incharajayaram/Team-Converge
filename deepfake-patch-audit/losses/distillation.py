"""Patch-level knowledge distillation loss: MSE (patches) + BCE (image)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDistillationLoss(nn.Module):
    """
    Patch-level distillation loss for student-teacher training.

    Combines:
    1. Patch-level MSE loss: MSE between aligned student and teacher patch logits
    2. Image-level BCE loss: Binary cross-entropy on pooled student prediction vs label

    Architecture:
    - Teacher output: (B, 1, 31, 31) patch-logit map
    - Student output: (B, 1, 126, 126) patch-logit map
    - Student aligned: (B, 1, 31, 31) via adaptive_avg_pool2d
    - Patch loss: MSE(student_aligned, teacher)
    - Task loss: BCE(pooled_student_logit, label)
    - Total: alpha_distill × patch_loss + alpha_task × task_loss
    
    SATURATION GUARD:
    - Normalizes extreme logits (>100 or <-100) to prevent gradient death
    - Centers and scales logits to reasonable range [-10, 10]
    """

    def __init__(self, alpha_distill=0.5, alpha_task=0.5, enable_saturation_guard=True, 
                 logit_clip_threshold=100.0):
        """
        Args:
            alpha_distill: Weight for patch-level MSE loss
            alpha_task: Weight for image-level BCE loss
            enable_saturation_guard: Whether to apply logit normalization
            logit_clip_threshold: Threshold for detecting saturated logits
        """
        super().__init__()
        self.alpha_distill = alpha_distill
        self.alpha_task = alpha_task
        self.enable_saturation_guard = enable_saturation_guard
        self.logit_clip_threshold = logit_clip_threshold
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Track saturation events for debugging
        self.saturation_count = 0

    def normalize_logits(self, logits, name="logits"):
        """
        Normalize extreme logits to prevent saturation.
        
        This is critical when using pretrained weights that may produce
        logits in the range of thousands, causing sigmoid(x) to saturate
        at exactly 1.0, leading to zero gradients.
        
        Strategy:
        1. Detect extreme values (|logit| > threshold)
        2. Center logits to zero mean
        3. Scale to std=1.0, then multiply by 5 (gives range ~[-15, 15])
        4. Clip to [-10, 10] for safety
        
        Args:
            logits: Tensor of any shape containing logits
            name: Name for logging
            
        Returns:
            Normalized logits in a trainable range
        """
        # Check for extreme values
        max_abs_logit = torch.max(torch.abs(logits))
        
        if max_abs_logit > self.logit_clip_threshold:
            self.saturation_count += 1
            
            if self.saturation_count <= 5:  # Log first 5 occurrences
                print(f"[WARN] Saturation detected in {name}: "
                      f"max_abs={max_abs_logit:.1f}, normalizing...")
            
            # Center to zero mean
            logits_normalized = logits - torch.mean(logits)
            
            # Scale to unit std, then multiply by 5 (gives ~3 std range)
            std = torch.std(logits_normalized) + 1e-8  # Avoid division by zero
            logits_normalized = (logits_normalized / std) * 5.0
            
            # Safety clip
            logits_normalized = torch.clamp(logits_normalized, -10.0, 10.0)
            
            return logits_normalized
        
        return logits

    def forward(self, student_patches, teacher_patches, student_image_logit, labels):
        """
        Compute total patch-level distillation loss.

        Args:
            student_patches: (B, 1, 126, 126) - student patch logits
            teacher_patches: (B, 1, 31, 31) - teacher patch logits
            student_image_logit: (B, 1) - pooled student prediction
            labels: (B,) - ground truth (0=real, 1=fake)

        Returns:
            total_loss: Combined distillation + task loss
            distill_loss: Patch-level MSE loss (for logging)
            task_loss: Image-level BCE loss (for logging)
        """
        # Verify shapes
        assert (
            student_patches.shape[1] == 1
        ), f"Expected channel dimension 1, got {student_patches.shape[1]}"
        assert (
            teacher_patches.shape[1] == 1
        ), f"Expected channel dimension 1, got {teacher_patches.shape[1]}"
        assert (
            student_image_logit.shape[1] == 1
        ), f"Expected output shape (B, 1), got {student_image_logit.shape}"

        # Apply saturation guard if enabled
        if self.enable_saturation_guard:
            student_patches = self.normalize_logits(student_patches, "student_patches")
            teacher_patches = self.normalize_logits(teacher_patches, "teacher_patches")
            student_image_logit = self.normalize_logits(student_image_logit, "student_image_logit")

        # Align patch maps: downsample student 126×126 → 31×31
        # This ensures spatial correspondence for patch-level MSE
        B, C, H_student, W_student = student_patches.shape
        H_teacher, W_teacher = teacher_patches.shape[2:]

        student_aligned = F.adaptive_avg_pool2d(student_patches, (H_teacher, W_teacher))

        # Patch-level loss: MSE on aligned patch logits (averaged per patch cell)
        # Reduction='mean' averages over all dimensions: batch × spatial × channel
        # This ensures patch loss is properly scaled relative to task loss
        distill_loss = F.mse_loss(student_aligned, teacher_patches, reduction='mean')

        # Image-level task loss: BCE on pooled logit vs label
        # Convert labels to float and add batch dimension for BCE
        labels_float = labels.float().unsqueeze(1)  # (B, 1)
        task_loss = self.bce_loss(student_image_logit, labels_float)

        # Combine losses
        total_loss = self.alpha_distill * distill_loss + self.alpha_task * task_loss

        return total_loss, distill_loss, task_loss

    def forward_with_details(self, student_patches, teacher_patches, student_image_logit, labels):
        """
        Compute loss and return component details.

        Returns:
            dict with 'total', 'distill', 'task' losses
        """
        total_loss, distill_loss, task_loss = self.forward(
            student_patches, teacher_patches, student_image_logit, labels
        )

        return {
            "total": total_loss,
            "distill": distill_loss,
            "task": task_loss,
        }
