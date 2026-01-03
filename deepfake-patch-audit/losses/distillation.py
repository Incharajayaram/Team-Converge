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
    """

    def __init__(self, alpha_distill=0.5, alpha_task=0.5):
        """
        Args:
            alpha_distill: Weight for patch-level MSE loss
            alpha_task: Weight for image-level BCE loss
        """
        super().__init__()
        self.alpha_distill = alpha_distill
        self.alpha_task = alpha_task
        self.bce_loss = nn.BCEWithLogitsLoss()

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
