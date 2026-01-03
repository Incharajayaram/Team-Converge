# Training Diagnostics Guide

Your training shows:
- **Total Loss: ~225** (looks huge, but it's dominated by distill loss)
- **Task Loss: ~0.7** (normal, near-random BCE)
- **AUC: ~0.50** (means model isn't learning classification)

## Problem Statement

The distill loss (~450) is overwhelming the task loss. The weighted sum is:
```
Total Loss = 0.5 × Distill_Loss(~450) + 0.5 × Task_Loss(~0.7)
           ≈ 225 + 0.35 = ~225
```

The model is learning to match teacher patches but **not learning to classify fake/real**.

## Root Cause Investigation

Run these diagnostics in order:

### Step 1: Check if Teacher is Informative

```bash
python scripts/diagnose_teacher.py
```

This will show:
- **Teacher AUC on your validation set**
- **Teacher patch logit statistics**

**Interpretation:**
- If `Teacher AUC < 0.55`: Teacher is not informative → **Domain shift problem**
- If `Teacher AUC > 0.70`: Teacher is good → **Loss scaling problem**

### Step 2: Check Logit Scale Mismatch

```bash
python scripts/diagnose_logits.py
```

This will show:
- Teacher patch logits: min, max, mean, std
- Student patch logits (aligned): min, max, mean, std
- MSE between them
- Scale ratio

**Interpretation:**
- If student and teacher ranges differ by 3x+: **Logit scale mismatch**
- If MSE is > 100: **Loss dominates, BCE signal is lost**

### Step 3: Test with BCE Only

Train without distillation to establish baseline:

```bash
# Modify the training script to use test_bce_only.yaml
# Or edit config/base.yaml and set:
# training:
#   distillation:
#     alpha_distill: 0.0   (disable distillation)
#     alpha_task: 1.0      (use only task loss)

python scripts/train_student_two_stage.py \
  --dataset-root dataset \
  --teacher-weights wildrf \
  --batch-size 32 \
  --epochs-s1 5 \
  --epochs-s2 20 \
  --device cuda
```

**Expected output if baseline works:**
```
Epoch 1: Loss: 0.6931 (near -ln(0.5), random guessing)
Epoch 5: Loss: 0.4500 (learning something)
Val AUC: > 0.60 (meaningful signal)
```

If BCE-only baseline doesn't work → **Dataset problem, not distillation**
If BCE-only baseline works well → **Distillation is the issue**

## Solutions by Root Cause

### ❌ Case 1: Teacher AUC < 0.55 (Domain Shift)

**Problem:** Teacher trained on different data than your dataset

**Solutions:**
1. **Fine-tune teacher first:**
   ```bash
   python scripts/train_student_two_stage.py \
     --disable-distillation \
     --epochs-s1 10 --epochs-s2 30
   ```
   Train teacher on your data (remove freeze_backbone)

2. **Use simpler baseline:**
   ```bash
   # Train student without teacher
   python scripts/train_student_two_stage.py --alpha-distill 0.0
   ```

3. **Check dataset:**
   - Are labels correct? (real = 0, fake = 1)
   - Is dataset balanced?
   - Do images look reasonable?

### ❌ Case 2: Student Range >> Teacher Range (Scale Mismatch)

**Problem:** Student outputs [-10, 10] while teacher outputs [-0.5, 0.5]

**Solutions:**
1. **Normalize before MSE:**
   ```python
   # In losses/distillation.py
   student_norm = (student_aligned - student_aligned.mean()) / (student_aligned.std() + 1e-8)
   teacher_norm = (teacher_patches - teacher_patches.mean()) / (teacher_patches.std() + 1e-8)
   distill_loss = F.mse_loss(student_norm, teacher_norm)
   ```

2. **Reduce alpha_distill:**
   ```yaml
   training:
     distillation:
       alpha_distill: 0.01  # Very small distillation weight
       alpha_task: 0.99     # Almost all task loss
   ```

3. **Add layer normalization:**
   - Normalize patch outputs in models before MSE

### ❌ Case 3: BCE-Only Baseline Also Fails (Data Problem)

**Problem:** Even without distillation, model can't learn

**Check:**
1. **Dataset split:**
   ```bash
   ls -lh dataset/train/real dataset/train/fake
   ls -lh dataset/val/real dataset/val/fake
   ```
   Are there enough samples?

2. **Label balance:**
   - Count: `ls dataset/train/real | wc -l`
   - Count: `ls dataset/train/fake | wc -l`
   - Should be roughly equal

3. **Image quality:**
   - Spot-check random images from train/fake and train/real
   - Are they actually different (not mislabeled)?

4. **Preprocessing:**
   - Check normalize_mean and normalize_std in config
   - Make sure image resizing works correctly

## Quick Fixes (Priority Order)

1. **Safest first:** Reduce alpha_distill
   ```yaml
   alpha_distill: 0.01
   alpha_task: 0.99
   ```
   Run training, check if AUC improves

2. **If #1 works:** Slowly increase distillation weight
   ```yaml
   alpha_distill: 0.05, 0.10, 0.20, ...
   ```

3. **If AUC still ~0.50:** Check teacher
   ```bash
   python scripts/diagnose_teacher.py
   ```

4. **If teacher AUC < 0.55:** Skip distillation for hackathon
   ```yaml
   alpha_distill: 0.0
   alpha_task: 1.0
   ```

## Summary Table

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Total Loss ~225, AUC ~0.50 | Distill dominates | Run diagnose_teacher.py | Reduce alpha_distill |
| Teacher AUC < 0.55 | Domain shift | Teacher not informative | Skip distillation |
| Student range >> Teacher | Scale mismatch | Huge MSE values | Normalize logits |
| BCE-only also fails | Data problem | Dataset issue | Fix train/val split |

## Example: Complete Diagnostic Flow

```bash
# 1. Test if teacher is good
python scripts/diagnose_teacher.py
# Output: "Teacher AUC: 0.72" → Good, distillation should work

# 2. Check logit scales
python scripts/diagnose_logits.py
# Output: "Scales similar, MSE ~0.5" → No mismatch

# 3. Reduce distillation weight
# Edit config/base.yaml:
#   alpha_distill: 0.1
#   alpha_task: 0.9

# 4. Train again
python scripts/train_student_two_stage.py ...
# Monitor: Does AUC improve?

# 5. If AUC improves, slowly increase distillation
#   alpha_distill: 0.2, 0.3, 0.5, ...
```

## Files Created

- `scripts/diagnose_teacher.py` - Check teacher AUC on your data
- `scripts/diagnose_logits.py` - Check logit scales
- `config/test_bce_only.yaml` - Config for BCE-only training

## Next Steps

1. **Run:** `python scripts/diagnose_teacher.py`
2. **Check output** for Teacher AUC
3. **Report back** with results
4. We'll decide if distillation is the issue or data is the issue
