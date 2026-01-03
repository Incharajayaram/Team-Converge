# Teacher Fine-tuning Workflow - Complete Implementation

## Overview
This document describes the complete teacher fine-tuning architecture and all changes made to support it.

## Architecture Stages

### Stage 0: Teacher Evaluation (Diagnostic)
**Script:** `scripts/diagnose_teacher.py` (Already existed - Fixed)

**What it does:**
- Loads pretrained teacher (WildRF_LaDeDa.pth or ForenSynth_LaDeDa.pth)
- Evaluates on validation set to check if teacher is informative
- Returns AUC score

**Status:** FIXED (patch MSE squeezing bug resolved)

```bash
python3 scripts/diagnose_teacher.py
```

**Output:** AUC score indicating if teacher needs fine-tuning
- AUC > 0.65: Teacher is good, skip fine-tuning
- AUC 0.55-0.65: Teacher is marginal, consider fine-tuning
- AUC < 0.55: Teacher is bad, MUST fine-tune

---

### Stage 1: Teacher Fine-tuning (NEW)
**Script:** `scripts/train_teacher.py` (NEW - Created)

**What it does:**
- Fine-tunes pretrained teacher on YOUR training set with BCE loss
- Unfreezes only last layers (default: layer1) to preserve pretrained knowledge
- Uses early stopping on validation AUC (patience=5)
- Saves best checkpoint to `weights/teacher/teacher_finetuned_best.pth`

**Configuration:**
- Loss: BCEWithLogitsLoss (combines sigmoid + BCE)
- Optimizer: Adam with lr=0.0001, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau on validation AUC
- Early stopping: patience=5 epochs

```bash
python3 scripts/train_teacher.py \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --patience 5 \
  --unfreeze-blocks 1
```

**Output:**
- Fine-tuned checkpoint: `weights/teacher/teacher_finetuned_best.pth`
- Training logs showing AUC progression

---

### Stage 2: Teacher Verification (NEW)
**Script:** `scripts/evaluate_teacher.py` (NEW - Created)

**What it does:**
- Loads fine-tuned teacher checkpoint
- Evaluates on validation set
- Provides guidance on whether to proceed with distillation

**Interpretation:**
- AUC >= 0.65: Ready for distillation
- AUC 0.55-0.65: Can proceed but results may be limited
- AUC < 0.55: Skip distillation, use BCE only

```bash
python3 scripts/evaluate_teacher.py \
  --checkpoint weights/teacher/teacher_finetuned_best.pth
```

**Output:**
- Teacher AUC on validation set
- Recommendation on whether to proceed with student training

---

### Stage 3: Student Training with Fixed Loss (UPDATED)
**Scripts:**
- `scripts/train_student.py` (UPDATED)
- `scripts/train_student_two_stage.py` (UPDATED)

**Changes Made:**

#### 1. Loss Function (`losses/distillation.py`)
```python
# OLD: distill_loss = F.mse_loss(student_aligned, teacher_patches)
# NEW: distill_loss = F.mse_loss(student_aligned, teacher_patches, reduction='mean')
```

**What changed:**
- Explicitly set `reduction='mean'` for patch MSE
- Averages loss per patch cell (not summed)
- Ensures distillation loss doesn't dominate task loss

#### 2. Configuration (`config/train.yaml`)
```yaml
# OLD values:
# alpha_distill: 0.5  # weight of patch MSE loss
# alpha_task: 0.5     # weight of image BCE loss

# NEW values:
alpha_distill: 0.05  # START SMALL
alpha_task: 0.95     # START HIGH
```

**Recommended progression:**
```yaml
# Step 1: Confirm BCE loss works first
alpha_distill: 0.01
alpha_task: 0.99

# Step 2: Introduce light distillation
alpha_distill: 0.05
alpha_task: 0.95

# Step 3: Increase distillation signal
alpha_distill: 0.1
alpha_task: 0.9

# Step 4: Balanced distillation
alpha_distill: 0.5
alpha_task: 0.5
```

#### 3. Training Scripts
Both `scripts/train_student.py` and `scripts/train_student_two_stage.py` now support:

```bash
--teacher-weights {wildrf|forensyth|finetuned}
```

**Usage:**

**Single-stage training with fine-tuned teacher:**
```bash
python3 scripts/train_student.py \
  --teacher-weights finetuned \
  --epochs 50 \
  --batch-size 16 \
  --lr 0.001
```

**Two-stage training with fine-tuned teacher:**
```bash
python3 scripts/train_student_two_stage.py \
  --teacher-weights finetuned \
  --epochs-s1 5 \
  --epochs-s2 20 \
  --lr-s1 0.001 \
  --lr-s2 0.0001
```

---

## Complete Workflow

### Step 1: Evaluate Pretrained Teacher
```bash
python3 scripts/diagnose_teacher.py
# Check AUC score
```

### Step 2: If AUC < 0.55, Fine-tune Teacher
```bash
python3 scripts/train_teacher.py
# Creates: weights/teacher/teacher_finetuned_best.pth
```

### Step 3: Verify Fine-tuned Teacher
```bash
python3 scripts/evaluate_teacher.py
# Check if AUC improved
```

### Step 4: Train Student with Fine-tuned Teacher
```bash
# Two-stage training (recommended)
python3 scripts/train_student_two_stage.py \
  --teacher-weights finetuned \
  --epochs-s1 5 \
  --epochs-s2 20

# OR single-stage training
python3 scripts/train_student.py \
  --teacher-weights finetuned \
  --epochs 50
```

### Step 5: Evaluate Student
```bash
python3 scripts/evaluate_student.py \
  --checkpoint outputs/checkpoints_two_stage/student_final.pt \
  --two-stage
```

---

## Files Changed

### New Files Created:
1. `scripts/train_teacher.py` - Teacher fine-tuning script
2. `scripts/evaluate_teacher.py` - Teacher evaluation script
3. `TEACHER_FINETUNING_WORKFLOW.md` - This documentation

### Files Updated:
1. `losses/distillation.py` - Explicit loss reduction
2. `config/train.yaml` - New default alpha values and progression guide
3. `scripts/train_student.py` - Added --teacher-weights option
4. `scripts/train_student_two_stage.py` - Added --teacher-weights option
5. `scripts/diagnose_teacher.py` - Fixed squeeze() bug

### Files Not Changed (Functioning as-is):
1. `training/train_student.py` - Trainer class (freezes teacher correctly)
2. `training/train_student_two_stage.py` - Trainer class (freezes teacher correctly)
3. `scripts/evaluate_student.py` - Student evaluation (no teacher involvement)

---

## Key Implementation Details

### Loss Scaling
The patch MSE loss is now properly averaged:
```python
# Reduction='mean' averages over all dimensions:
# batch × spatial_height × spatial_width × channel
# This ensures the loss is in the right scale
```

### Teacher Freezing
Both trainer classes freeze the teacher at initialization:
```python
# In __init__:
self.teacher_model.eval()
for param in self.teacher_model.parameters():
    param.requires_grad = False
```

This is correct and ensures the frozen teacher (fine-tuned or pretrained) doesn't update during student training.

### Alpha Weight Progression
The default config now starts with small distillation:
- Initial: `alpha_distill=0.05, alpha_task=0.95`
- Rationale: Confirm BCE loss learns first, then gradually increase distillation signal
- User can adjust in config or during training

---

## Troubleshooting

### Issue: Teacher fine-tuning isn't improving AUC
**Solutions:**
1. Increase epochs: `--epochs 100`
2. Increase learning rate: `--lr 0.0005` (higher than default 0.0001)
3. Unfreeze more blocks: `--unfreeze-blocks 2`
4. Check dataset labels: Run `scripts/analyze_dataset.py` if available

### Issue: Student training loss is very high
**Solutions:**
1. Start with even smaller alpha_distill: `alpha_distill: 0.01`
2. Verify fine-tuned teacher AUC is good: `python3 scripts/evaluate_teacher.py`
3. Check that teacher checkpoint is being loaded correctly

### Issue: Student AUC doesn't improve with distillation
**Explanation:** This can happen if teacher signal is weak
**Solutions:**
1. Increase alpha_distill gradually
2. Train more epochs
3. Consider skipping distillation and using BCE-only training

---

## Summary of Improvements

✅ **Proper loss scaling:** Patch MSE is averaged, not summed
✅ **Flexible alpha weights:** Start small, increase gradually
✅ **Teacher fine-tuning:** Support for adapting teacher to your data
✅ **Checkpoint management:** Easy switching between pretrained and fine-tuned
✅ **Clear validation gates:** Teacher must pass AUC threshold
✅ **Progressive learning:** Recommended alpha progression path

---

## Next Steps

1. Run Stage 0: `python3 scripts/diagnose_teacher.py`
2. If AUC < 0.55, run Stage 1: `python3 scripts/train_teacher.py`
3. Run Stage 2: `python3 scripts/evaluate_teacher.py`
4. If approved, run Stage 3: `python3 scripts/train_student_two_stage.py --teacher-weights finetuned`
5. Evaluate: `python3 scripts/evaluate_student.py --two-stage`
