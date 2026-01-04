#!/usr/bin/env python3
"""
Deterministic Verification Experiments for Hyperparameter Choices.

This script verifies:
1. Saturation guard correctly normalizes extreme logits
2. Alpha weights (0.1/0.9) produce balanced loss contributions
3. Gradient flow is maintained after normalization
4. Loss values are in expected ranges
5. Model outputs are deterministic with seed

Run this BEFORE full training to validate fixes are working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def set_seed(seed=42):
    """Ensure deterministic behavior."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def test_saturation_guard():
    """
    Test 1: Verify saturation guard correctly handles extreme logits.
    
    Expected behavior:
    - Logits with |value| > 100 are normalized to [-10, 10] range
    - Normalized logits produce non-zero gradients through sigmoid
    """
    print("\n" + "=" * 70)
    print("TEST 1: Saturation Guard Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    loss_fn = PatchDistillationLoss(
        alpha_distill=0.1, 
        alpha_task=0.9,
        enable_saturation_guard=True,
        logit_clip_threshold=100.0
    )
    
    # Simulate extreme logits (like ForenSynth pretrained weights)
    extreme_logits = torch.tensor([[-5000.0], [9700.0], [-8500.0], [7200.0]])
    
    # Normalize
    normalized = loss_fn.normalize_logits(extreme_logits.clone(), "test_extreme")
    
    # Verify range
    max_abs = torch.max(torch.abs(normalized)).item()
    mean_val = torch.mean(normalized).item()
    
    print(f"  Input range: [{extreme_logits.min():.1f}, {extreme_logits.max():.1f}]")
    print(f"  Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print(f"  Output max_abs: {max_abs:.3f}")
    print(f"  Output mean: {mean_val:.6f}")
    
    # Check assertions
    assert max_abs <= 10.0, f"Max abs should be <= 10, got {max_abs}"
    assert abs(mean_val) < 1e-4, f"Mean should be ~0, got {mean_val}"
    
    # Verify gradients flow
    normalized_grad = normalized.clone().requires_grad_(True)
    sigmoid_out = torch.sigmoid(normalized_grad)
    loss = sigmoid_out.mean()
    loss.backward()
    
    grad_norm = normalized_grad.grad.norm().item()
    print(f"  Gradient norm after sigmoid: {grad_norm:.6f}")
    
    assert grad_norm > 1e-6, f"Gradient should be non-zero, got {grad_norm}"
    
    print("  [PASS] Saturation guard PASSED")
    return True


def test_alpha_balance():
    """
    Test 2: Verify alpha weights produce balanced loss contributions.
    
    With alpha_distill=0.1 and alpha_task=0.9:
    - Task loss should dominate total loss (~90%)
    - Distillation loss contributes ~10%
    """
    print("\n" + "=" * 70)
    print("TEST 2: Alpha Weight Balance Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    loss_fn = PatchDistillationLoss(
        alpha_distill=0.1,
        alpha_task=0.9,
        enable_saturation_guard=True
    )
    
    # Create synthetic data
    batch_size = 4
    student_patches = torch.randn(batch_size, 1, 126, 126) * 2  # Normal range
    teacher_patches = torch.randn(batch_size, 1, 31, 31) * 2
    student_image_logit = torch.randn(batch_size, 1)
    labels = torch.tensor([0, 1, 0, 1])  # Binary labels
    
    # Compute losses
    total_loss, distill_loss, task_loss = loss_fn(
        student_patches, teacher_patches, student_image_logit, labels
    )
    
    # Calculate contributions
    distill_contrib = (0.1 * distill_loss.item())
    task_contrib = (0.9 * task_loss.item())
    total_val = total_loss.item()
    
    print(f"  Distill loss (raw): {distill_loss.item():.4f}")
    print(f"  Task loss (raw): {task_loss.item():.4f}")
    print(f"  Distill contribution (0.1×): {distill_contrib:.4f}")
    print(f"  Task contribution (0.9×): {task_contrib:.4f}")
    print(f"  Total loss: {total_val:.4f}")
    
    # Verify task dominates
    task_ratio = task_contrib / (total_val + 1e-8)
    print(f"  Task ratio in total: {task_ratio*100:.1f}%")
    
    # Task should be ~90% of total (allow some tolerance)
    assert task_ratio > 0.6, f"Task ratio should be > 60%, got {task_ratio*100:.1f}%"
    
    print("  [PASS] Alpha balance PASSED")
    return True


def test_gradient_flow():
    """
    Test 3: Verify gradients flow through the entire loss computation.
    
    This ensures the model can actually learn.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Flow Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    loss_fn = PatchDistillationLoss(
        alpha_distill=0.1,
        alpha_task=0.9,
        enable_saturation_guard=True
    )
    
    # Create synthetic data with gradients
    batch_size = 4
    student_patches = torch.randn(batch_size, 1, 126, 126, requires_grad=True)
    teacher_patches = torch.randn(batch_size, 1, 31, 31)
    student_image_logit = torch.randn(batch_size, 1, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    
    # Forward pass
    total_loss, _, _ = loss_fn(
        student_patches, teacher_patches, student_image_logit, labels
    )
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients exist and are non-zero
    patches_grad_norm = student_patches.grad.norm().item()
    logit_grad_norm = student_image_logit.grad.norm().item()
    
    print(f"  Patch gradients norm: {patches_grad_norm:.6f}")
    print(f"  Logit gradients norm: {logit_grad_norm:.6f}")
    
    assert patches_grad_norm > 1e-8, f"Patch gradients should be non-zero"
    assert logit_grad_norm > 1e-8, f"Logit gradients should be non-zero"
    
    print("  [PASS] Gradient flow PASSED")
    return True


def test_loss_ranges():
    """
    Test 4: Verify loss values are in expected ranges.
    
    For BCE with random predictions:
    - Expected loss ~0.69 (ln(2))
    
    For MSE with normalized logits:
    - Expected loss depends on alignment quality
    """
    print("\n" + "=" * 70)
    print("TEST 4: Loss Range Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    loss_fn = PatchDistillationLoss(
        alpha_distill=0.1,
        alpha_task=0.9,
        enable_saturation_guard=True
    )
    
    # Run multiple times and check consistency
    losses = []
    for i in range(5):
        set_seed(42 + i)
        batch_size = 8
        student_patches = torch.randn(batch_size, 1, 126, 126) * 3
        teacher_patches = torch.randn(batch_size, 1, 31, 31) * 3
        student_image_logit = torch.randn(batch_size, 1)
        labels = torch.randint(0, 2, (batch_size,))
        
        total_loss, distill_loss, task_loss = loss_fn(
            student_patches, teacher_patches, student_image_logit, labels
        )
        losses.append({
            'total': total_loss.item(),
            'distill': distill_loss.item(),
            'task': task_loss.item()
        })
    
    avg_total = np.mean([l['total'] for l in losses])
    avg_task = np.mean([l['task'] for l in losses])
    
    print(f"  Average total loss: {avg_total:.4f}")
    print(f"  Average task loss: {avg_task:.4f}")
    print(f"  Expected BCE (random): ~0.69 (ln(2))")
    
    # BCE with random predictions should be around ln(2) ≈ 0.693
    assert 0.3 < avg_task < 1.5, f"Task loss should be ~0.69, got {avg_task:.4f}"
    assert avg_total < 10.0, f"Total loss should be < 10, got {avg_total:.4f}"
    
    print("  [PASS] Loss ranges PASSED")
    return True


def test_extreme_saturation_recovery():
    """
    Test 5: Verify extreme logits (9700+) produce trainable gradients.
    
    This is the critical test for the saturation fix.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Extreme Saturation Recovery Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    # Test WITHOUT saturation guard (should fail)
    print("  Testing WITHOUT saturation guard...")
    loss_fn_no_guard = PatchDistillationLoss(
        alpha_distill=0.1,
        alpha_task=0.9,
        enable_saturation_guard=False  # Disabled
    )
    
    batch_size = 4
    # Extreme logits like what ForenSynth produces
    extreme_student = torch.ones(batch_size, 1, 126, 126) * 9700
    extreme_teacher = torch.ones(batch_size, 1, 31, 31) * 8000
    extreme_logit = torch.ones(batch_size, 1) * 9000
    extreme_logit.requires_grad = True
    labels = torch.tensor([0, 1, 0, 1])
    
    total_no_guard, _, _ = loss_fn_no_guard(
        extreme_student, extreme_teacher, extreme_logit, labels
    )
    total_no_guard.backward()
    
    grad_no_guard = extreme_logit.grad.norm().item()
    print(f"    Gradient norm (no guard): {grad_no_guard:.10f}")
    
    # Test WITH saturation guard (should work)
    print("  Testing WITH saturation guard...")
    loss_fn_with_guard = PatchDistillationLoss(
        alpha_distill=0.1,
        alpha_task=0.9,
        enable_saturation_guard=True  # Enabled
    )
    
    extreme_logit2 = torch.ones(batch_size, 1) * 9000
    extreme_logit2.requires_grad = True
    
    total_with_guard, _, _ = loss_fn_with_guard(
        extreme_student.clone(), extreme_teacher.clone(), extreme_logit2, labels
    )
    total_with_guard.backward()
    
    grad_with_guard = extreme_logit2.grad.norm().item()
    print(f"    Gradient norm (with guard): {grad_with_guard:.10f}")
    
    # With guard should have MUCH larger gradients
    print(f"    Improvement factor: {grad_with_guard / (grad_no_guard + 1e-20):.1f}x")
    
    assert grad_with_guard > grad_no_guard * 1000, \
        f"Guard should improve gradients by >1000x, got {grad_with_guard / (grad_no_guard + 1e-20):.1f}x"
    
    print("  [PASS] Saturation recovery PASSED")
    return True


def test_determinism():
    """
    Test 6: Verify training is deterministic with seed.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Determinism Verification")
    print("=" * 70)
    
    from losses.distillation import PatchDistillationLoss
    
    results = []
    for run in range(2):
        set_seed(42)  # Same seed
        
        loss_fn = PatchDistillationLoss(
            alpha_distill=0.1,
            alpha_task=0.9,
            enable_saturation_guard=True
        )
        
        batch_size = 4
        student_patches = torch.randn(batch_size, 1, 126, 126)
        teacher_patches = torch.randn(batch_size, 1, 31, 31)
        student_image_logit = torch.randn(batch_size, 1)
        labels = torch.tensor([0, 1, 0, 1])
        
        total_loss, _, _ = loss_fn(
            student_patches, teacher_patches, student_image_logit, labels
        )
        results.append(total_loss.item())
    
    print(f"  Run 1 loss: {results[0]:.10f}")
    print(f"  Run 2 loss: {results[1]:.10f}")
    print(f"  Difference: {abs(results[0] - results[1]):.15f}")
    
    assert abs(results[0] - results[1]) < 1e-6, \
        f"Results should be identical with same seed"
    
    print("  [PASS] Determinism PASSED")
    return True


def main():
    print("\n" + "=" * 70)
    print("  HYPERPARAMETER VERIFICATION EXPERIMENTS")
    print("  Running deterministic tests to validate choices")
    print("=" * 70)
    
    tests = [
        ("Saturation Guard", test_saturation_guard),
        ("Alpha Balance", test_alpha_balance),
        ("Gradient Flow", test_gradient_flow),
        ("Loss Ranges", test_loss_ranges),
        ("Extreme Saturation Recovery", test_extreme_saturation_recovery),
        ("Determinism", test_determinism),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n  [PASS] ALL TESTS PASSED - Hyperparameters are valid!")
        print("  Safe to proceed with full training.")
    else:
        print(f"\n  [FAIL] {failed} TESTS FAILED - Review hyperparameters!")
        sys.exit(1)


if __name__ == "__main__":
    main()
