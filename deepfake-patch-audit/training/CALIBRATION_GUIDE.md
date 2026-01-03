# Threshold Calibration & Diagnostics Guide

## Overview

After training your student model with two-stage distillation, the `ThresholdCalibrator` automatically:

1. **Runs inference** on the validation set
2. **Computes ROC-AUC** to evaluate overall discrimination ability
3. **Finds optimal threshold** (`t_star`) that maximizes accuracy on validation data
4. **Generates diagnostics** including confusion matrix, precision/recall, and class-wise statistics
5. **Saves results** to JSON and CSV for further analysis

## What Happens Automatically

### During Training

When you run two-stage training:
```bash
python scripts/train_student_two_stage.py
```

At the end of training, **calibration runs automatically**:
- Loads the final trained student model
- Runs inference on the entire validation set
- Computes optimal decision threshold
- Saves results to: `outputs/checkpoints_two_stage/calibration/`

### Output Files

After training completes, you'll find in `outputs/checkpoints_two_stage/calibration/`:

```
calibration/
├── threshold_calibration.json    # All calibration results
├── predictions.csv               # [index, y_true, p_fake, y_pred@t_star]
└── (training_history_two_stage.json will also include calibration results)
```

## Understanding the Results

### threshold_calibration.json

```json
{
  "auc": 0.9876,
  "optimal_threshold": 0.48,
  "diagnostics": {
    "threshold": 0.48,
    "metrics": {
      "accuracy": 0.9234,
      "precision": 0.9150,
      "recall": 0.9320,
      "f1": 0.9235,
      "fpr": 0.0089
    },
    "confusion_matrix": {
      "TP": 234,
      "TN": 987,
      "FP": 9,
      "FN": 12
    },
    "class_statistics": {
      "real": {
        "count": 996,
        "p_fake_mean": 0.1234,
        "p_fake_std": 0.0856,
        "p_fake_min": 0.0012,
        "p_fake_max": 0.4567
      },
      "fake": {
        "count": 246,
        "p_fake_mean": 0.8765,
        "p_fake_std": 0.0943,
        "p_fake_min": 0.5432,
        "p_fake_max": 0.9987
      }
    }
  },
  "histogram": {
    "bins": 10,
    "edges": [0.0, 0.1, 0.2, ...],
    "real": [45, 67, 89, ...],
    "fake": [1, 2, 3, ..., 98]
  },
  "threshold_analysis": {
    "0.0": {"accuracy": 0.8234, "fpr": 1.0, ...},
    "0.01": {"accuracy": 0.8245, "fpr": 0.9876, ...},
    ...
    "0.48": {"accuracy": 0.9234, "fpr": 0.0089, ...}
  }
}
```

### Key Metrics Explained

- **AUC**: Overall discriminative ability (0.0-1.0, higher is better)
- **t_star**: Optimal decision threshold for classification
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) — false positive rate among positives
- **Recall**: TP / (TP + FN) — true positive rate
- **F1**: Harmonic mean of precision and recall
- **FPR**: FP / (FP + TN) — false positive rate on real samples

### Class Statistics

- **Real class (y_true=0)**: Statistics for non-fake (genuine) images
  - Should have LOW mean p_fake (ideally < 0.2)
  - Shows model's confidence in negative class

- **Fake class (y_true=1)**: Statistics for fake images
  - Should have HIGH mean p_fake (ideally > 0.8)
  - Shows model's confidence in positive class

If there's significant overlap (e.g., real mean = 0.6, fake mean = 0.7), the model has low discriminative ability.

## Using the Calibrated Threshold

### Automatic in Evaluation Script

The evaluation script automatically loads and uses `t_star`:

```bash
python scripts/evaluate_student.py
```

Output will show:
```
✓ Loaded calibrated threshold: t_star = 0.48
Accuracy:  0.9234  (at t_star = 0.48)
Precision: 0.9150
Recall:    0.9320
F1-Score:  0.9235
```

### Manual Usage in Your Code

```python
import json
from pathlib import Path

# Load calibration
calibration_file = Path("outputs/checkpoints_two_stage/calibration/threshold_calibration.json")
with open(calibration_file) as f:
    cal_data = json.load(f)

t_star = cal_data["optimal_threshold"]  # e.g., 0.48

# Use in inference
predictions = (probabilities > t_star).astype(int)
```

## Why Different from 0.5?

The default threshold of 0.5 assumes equal costs for false positives and false negatives. However:

- **If false positives are costly** (e.g., wrongly flagging real as fake): Lower t_star (e.g., 0.42)
- **If false negatives are costly** (e.g., missing real fakes): Higher t_star (e.g., 0.54)

The calibrator chooses t_star that **maximizes accuracy on validation set**, with ties broken by **minimum FPR** (to avoid false positives).

## Threshold Analysis

The `threshold_analysis` object shows accuracy and FPR for every threshold from 0.00 to 1.00:

```json
"threshold_analysis": {
  "0.40": {"accuracy": 0.9212, "fpr": 0.0145},
  "0.45": {"accuracy": 0.9228, "fpr": 0.0112},
  "0.48": {"accuracy": 0.9234, "fpr": 0.0089},  ← Best
  "0.50": {"accuracy": 0.9230, "fpr": 0.0098},
  "0.55": {"accuracy": 0.9210, "fpr": 0.0065}
}
```

Use this to understand the accuracy-FPR trade-off and choose t_star based on your application needs.

## Predictions CSV

`predictions.csv` contains:
```
index,y_true,p_fake,y_pred@0.48
0,0,0.1234,0
1,1,0.8765,1
2,0,0.4512,0
...
```

Use this for:
- Finding misclassified samples
- Debugging edge cases near the decision boundary
- Understanding which samples are "confident" vs "uncertain"

## For Quantized Models

After quantization, run calibration again with the quantized model:

```python
quantized_model = load_quantized_model(...)
calibrator = ThresholdCalibrator(
    model=quantized_model,
    val_loader=val_loader,
    pooling=pooling,
    device=device,
    output_dir="outputs/checkpoints_two_stage/calibration_quantized"
)
t_star_quant, results_quant = calibrator.calibrate(suffix="_quantized")
```

This creates:
- `threshold_calibration_quantized.json`
- `predictions_quantized.csv`

Compare with non-quantized t_star to assess quantization impact.

## Troubleshooting

### Issue: t_star very different from 0.5

**Cause**: The model is miscalibrated (outputs don't match true probabilities)

**Solution**:
1. Check class balance in validation set
2. Review class statistics — overlap indicates poor discrimination
3. Consider longer training or better data

### Issue: High AUC but low accuracy at t_star

**Cause**: Model separates real/fake well but outputs are skewed

**Solution**: This is normal if one class is much larger. Check confusion matrix for practical impact.

### Issue: File not found during evaluation

**Cause**: Calibration didn't run or saved to different location

**Solution**:
```bash
ls outputs/checkpoints_two_stage/calibration/
```

Check the output directory exists and contains JSON file.

## Summary

| Component | Purpose | Location |
|-----------|---------|----------|
| `threshold_calibration.json` | All calibration results | `calibration/` |
| `predictions.csv` | Per-sample predictions | `calibration/` |
| `t_star` value | Optimal threshold | Used in eval script |
| Class statistics | Discrimination quality | `diagnostics/` |
| Threshold analysis | Accuracy-FPR trade-off | `threshold_analysis/` |

The calibration is **production-ready**: use `t_star` for all downstream decisions instead of default 0.5.
