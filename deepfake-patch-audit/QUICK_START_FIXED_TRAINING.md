# Quick Start: Fixed Student Model Training

## What Was Fixed

All critical training issues have been resolved:

✅ **Scale Mismatch** - Adaptive scale normalization in loss
✅ **Exploding Loss** - Reduced alpha_distill, gradient clipping, KL divergence
✅ **Mode Collapse** - Variance regularization, task loss emphasis
✅ **AUC Stuck at 0.5** - Better loss balancing, proper initialization
✅ **Windows Issues** - num_workers=0, proper memory management
✅ **Training Instability** - Warmup, better schedulers, layer-wise LR

---

## Quick Start (5 Minutes)

### Step 1: Create Training Script

Create `train_student_fixed.py`:

```python
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Import improved components
from losses.distillation_improved import ImprovedPatchDistillationLoss, ModeCollapsePrevention
from training.train_student_improved import ImprovedTwoStagePatchStudentTrainer
from pooling import TopKLogitPooling
from models import TinyLaDeDa, LaDeDa9

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    print("Loading models...")
    teacher = LaDeDa9(pretrained=True).to(device)
    student = TinyLaDeDa().to(device)

    # Setup data loaders
    # CRITICAL FOR WINDOWS: num_workers=0
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,  # ← Windows compatibility
        pin_memory=False,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle=False
    )

    # Setup improved loss
    criterion = ImprovedPatchDistillationLoss(
        alpha_distill=0.3,  # REDUCED from 0.5
        alpha_task=0.7,     # INCREASED from 0.5
        temperature=4.0,    # Softer targets
        use_kl_loss=True,   # Scale-invariant
        enable_scale_matching=True  # Fix scale mismatch
    )

    # Setup pooling
    pooling = TopKLogitPooling(k=10, device=device)

    # Setup trainer with all fixes
    trainer = ImprovedTwoStagePatchStudentTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=device,
        # Stage 1: Classifier training
        stage1_epochs=5,
        stage1_lr=0.0001,  # REDUCED from 0.001
        stage1_warmup_epochs=0.5,
        # Stage 2: Fine-tuning
        stage2_epochs=20,
        stage2_lr=0.00005,  # REDUCED from 0.0001
        stage2_backbone_lr=0.000005,  # Layer-wise LR
        stage2_warmup_epochs=1.0,
        # Gradient control
        gradient_clip_norm=0.5,  # Prevent explosion
        gradient_clip_value=1.0,
        # Checkpointing
        checkpoint_dir="outputs/checkpoints",
        save_best_only=True
    )

    # Train with all fixes applied!
    trainer.train()

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation AUC: {trainer.best_val_auc:.4f}")
    print(f"Best checkpoint: {trainer.best_checkpoint_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

### Step 2: Run Training

```bash
python train_student_fixed.py
```

### Step 3: Monitor Training

Watch for these signs that training is working correctly:

```
✅ GOOD SIGNS:
- Loss smooth, decreasing over time (not exploding)
- Variance > 1e-4 (not collapsing to constant)
- AUC increasing from 0.5 toward 0.7+
- Gradient norms stable (0.1 - 1.0)

❌ BAD SIGNS:
- Loss jumps to 100+ in one batch → reduce LR
- Variance < 1e-4 → mode collapse detected
- AUC stays at 0.5 → increase alpha_task
- Gradient norms > 10 → check gradient clipping
```

---

## Configuration Guide

### If Training is Too Slow

Reduce safety margins:

```python
trainer = ImprovedTwoStagePatchStudentTrainer(
    ...
    stage1_lr=0.0001,  # Keep safe
    stage2_lr=0.00005,
    gradient_clip_norm=0.5,  # Keep for stability
    stage1_warmup_epochs=0,  # Reduce warmup
    stage2_warmup_epochs=0.5
)
```

### If Loss Explodes

Increase safety:

```python
trainer = ImprovedTwoStagePatchStudentTrainer(
    ...
    stage1_lr=0.00001,  # 10x smaller
    stage2_lr=0.000005,
    gradient_clip_norm=0.2,  # Tighter clipping
)
```

### If Mode Collapse (variance → 0)

Increase task loss:

```python
criterion = ImprovedPatchDistillationLoss(
    alpha_distill=0.1,  # Even lower
    alpha_task=0.9,     # Emphasize task
    temperature=4.0
)
```

### If AUC Stuck at 0.5

Definitely increase task loss:

```python
criterion = ImprovedPatchDistillationLoss(
    alpha_distill=0.05,  # Minimal distillation
    alpha_task=0.95,     # Almost pure task loss
    temperature=4.0
)
```

---

## Performance Expectations

### Timeline
- **Stage 1** (Classifier): 5 epochs, ~30 min on GPU
- **Stage 2** (Fine-tuning): 20 epochs, ~2-3 hours on GPU
- **Total**: ~3 hours to full convergence

### Expected Results
- **Early epochs (1-5)**: Loss decreases, AUC rises from 0.5
- **Mid training (6-15)**: AUC plateaus (normal)
- **Late training (15+)**: Fine-tuning, incremental AUC gains
- **Final**: AUC ≈ 0.8-0.85 (depending on teacher quality)

### Key Metrics to Track

```python
# In training loop:
epoch_metrics = {
    'train_loss': 4.2,
    'train_variance': 0.15,  # Should stay > 1e-4
    'train_distill_loss': 1.2,
    'train_task_loss': 3.0,
    'gradient_norm': 0.42,
    'val_auc': 0.72,
    'learning_rate': 0.0001
}
```

---

## Common Issues & Fixes

### Issue: "Loss explosion detected: 125.4 > 100.0"

**Fix:**
```python
# Reduce learning rate
trainer.stage1_lr = 0.00001  # 10x reduction
trainer.stage2_lr = 0.000005

# Or increase gradient clipping
trainer.gradient_clip_norm = 0.2
```

### Issue: "Mode collapse detected! Variance=1.23e-08"

**Fix:**
```python
# Strongly emphasize task loss
criterion = ImprovedPatchDistillationLoss(
    alpha_distill=0.05,
    alpha_task=0.95
)
```

### Issue: "AUC stuck at 0.5, no improvement for 10 epochs"

**Fix:**
```python
# Check if task loss is being used
print(f"alpha_task = {criterion.alpha_task}")  # Should be 0.7+

# If still stuck, make it even stronger
criterion.alpha_task = 0.95
criterion.alpha_distill = 0.05
```

### Issue: "Windows crash with DataLoader"

**Fix:**
```python
# Always use these settings on Windows:
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=0,  # ← CRITICAL
    pin_memory=False,  # ← Disable
    shuffle=True
)
```

---

## Monitoring Dashboard

Add this to your training loop for real-time monitoring:

```python
import json
from pathlib import Path

def save_training_metrics(metrics, output_path="outputs/metrics.json"):
    """Save metrics for monitoring."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

# In training loop:
for epoch in range(num_epochs):
    metrics = {
        'epoch': epoch,
        'train_loss': trainer.history['train_loss'][-1],
        'train_variance': trainer.history['train_std'][-1],
        'val_auc': trainer.history['val_auc'][-1],
        'learning_rate': trainer.optimizer.param_groups[0]['lr'],
        'gradient_norm': trainer.history['gradient_norm'][-1]
    }
    save_training_metrics(metrics)
```

Then monitor with:

```bash
watch -n 5 'cat outputs/metrics.json'  # Linux/Mac
type outputs\metrics.json  # Windows (every 5 seconds)
```

---

## Validation Checklist

Before running full training, verify setup:

- [ ] Can load teacher model
- [ ] Can load student model
- [ ] DataLoader works (batch returns images and labels)
- [ ] Forward pass works: `student(images) → (B, 1, 126, 126)`
- [ ] Teacher forward: `teacher(images) → (B, 1, 31, 31)`
- [ ] Loss computation works without NaN/Inf
- [ ] Optimizer step works
- [ ] Checkpoint saving works
- [ ] num_workers=0 (especially on Windows)
- [ ] Gradient clipping enabled

---

## Next Steps

1. **Run training** with the fixed configuration
2. **Monitor** loss, variance, and AUC in first epoch
3. **Adjust** learning rates if needed
4. **Save** best checkpoint when validation AUC peaks
5. **Export** to ONNX for deployment

---

## Support

If you encounter issues:

1. Check `TRAINING_FIXES.md` for detailed explanations
2. Review `config_training_improved.yaml` for examples
3. Enable all monitoring (`enable_diagnostics=True`)
4. Check gradient norms and loss explosions in logs

Good luck! 🚀
