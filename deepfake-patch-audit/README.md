# Deepfake Patch-Audit

Deepfake detection using patch-level teacher-student distillation with quantization auditing.

## Overview

This project implements a scalable deepfake detection system based on:
- **Teacher-Student Architecture**: A large pretrained LaDeDa teacher model distills knowledge to a lightweight TinyLaDeDa student
- **Patch-based Analysis**: Processes images as patches to identify localized deepfake artifacts
- **Top-K Pooling**: Aggregates patch-level predictions for final classification
- **Quantization Auditing**: Rigorous testing of quantized models for edge deployment

## Project Structure

```
deepfake-patch-audit/
├── config/                 # Configuration files
│   ├── base.yaml          # Architecture contract
│   ├── dataset.yaml       # Dataset paths and splits
│   ├── train.yaml         # Training hyperparameters
│   └── quant.yaml         # Quantization settings
│
├── data/                  # Data directory
│   ├── splits/            # CSV files for train/val/test
│   └── samples/           # Debug sample images
│
├── datasets/              # Data loading
│   ├── base_dataset.py    # Base dataset class
│   ├── frame_dataset.py   # Frame-based loader
│   └── transforms.py      # Preprocessing (normalize, JPEG-shift)
│
├── models/                # Model implementations
│   ├── teacher/
│   │   ├── ladeda_wrapper.py    # Pretrained LaDeDa
│   │   └── patch_adapter.py     # Patch grid enforcement
│   │
│   ├── student/
│   │   ├── tiny_ladeda.py       # Lightweight student
│   │   └── blocks.py            # Conv/Bottleneck blocks
│   │
│   └── pooling.py               # Top-K aggregation
│
├── losses/                # Training losses
│   └── distillation.py   # KD loss (MSE + BCE)
│
├── inference/             # Prediction pipeline
│   ├── pipeline.py       # Main inference
│   └── heatmap.py        # Patch visualization
│
├── training/              # Training utilities
│   ├── train_student.py  # Student training loop
│   └── eval_loop.py      # Validation/testing
│
├── evaluation/            # Metrics and analysis
│   ├── metrics.py        # AUC, Accuracy@τ
│   ├── threshold.py      # Threshold tuning
│   └── distribution_shift.py  # JPEG robustness tests
│
├── quantization/          # Quantization tools
│   ├── dynamic_range.py  # Preferred quantization
│   ├── full_int8.py      # Optional (harder)
│   └── audit.py          # Float vs quant comparison
│
├── federated/             # Federated learning (boxed extension)
│   ├── client.py         # Client trainer
│   ├── server.py         # FedProx aggregator
│   └── simulation.py     # Federated simulation
│
├── scripts/               # Utility scripts
│   ├── audit_teacher.py     # Verify patch alignment
│   ├── audit_student.py     # Check student structure
│   ├── export_tflite.py     # Export to TFLite
│   └── run_eval.py          # One-command evaluation
│
├── tests/                 # Unit tests
│   ├── test_patch_alignment.py
│   ├── test_topk_pooling.py
│   ├── test_quant_consistency.py
│   └── test_inference_contract.py
│
├── outputs/               # Results directory
│   ├── checkpoints/       # Model weights
│   ├── logs/              # Training logs
│   └── heatmaps/          # Visualizations
│
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Complete Training Pipeline

### Stage 0: Evaluate Pretrained Teacher

Check if the pretrained teacher is informative on your dataset:

```bash
python scripts/diagnose_teacher.py
# Output: Teacher AUC on validation set
```

**Interpretation:**
- AUC > 0.65: Teacher is good → skip fine-tuning, proceed to student training
- AUC 0.55-0.65: Teacher is marginal → consider fine-tuning
- AUC < 0.55: Teacher is bad → **MUST fine-tune** or skip distillation

### Stage 1: Fine-tune Teacher (if needed)

If teacher AUC < 0.55, fine-tune it on your training data:

```bash
python scripts/train_teacher.py \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --patience 5 \
  --device cuda
# Output: weights/teacher/teacher_finetuned_best.pth
```

**Features:**
- Unfreezes only last layers (preserves pretrained knowledge)
- BCE loss for binary classification
- Early stopping on validation AUC
- Saves best checkpoint

### Stage 2: Verify Fine-tuned Teacher

Evaluate the fine-tuned teacher:

```bash
python scripts/evaluate_teacher.py \
  --checkpoint weights/teacher/teacher_finetuned_best.pth
# Output: Teacher AUC and distillation readiness assessment
```

### Stage 3: Train Student with Distillation

Train student model using fine-tuned or pretrained teacher:

```bash
# Two-stage training (recommended)
python scripts/train_student_two_stage.py \
  --dataset-root dataset \
  --teacher-weights finetuned \  # or 'wildrf', 'forensyth'
  --batch-size 32 \
  --epochs-s1 5 \
  --epochs-s2 20 \
  --lr-s1 0.001 \
  --lr-s2 0.0001 \
  --device cuda

# OR single-stage training
python scripts/train_student.py \
  --dataset-root dataset \
  --teacher-weights finetuned \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --device cuda
```

**Key Points:**
- Default alpha values start small (0.05 distill / 0.95 task) for stable learning
- Patch MSE is averaged per patch cell (not summed) for proper loss scaling
- Two-stage training: Stage 1 trains classifier, Stage 2 fine-tunes backbone

### Stage 4: Evaluate Student

Evaluate the trained student on test set:

```bash
# Single-stage
python scripts/evaluate_student.py \
  --checkpoint outputs/checkpoints/student_final.pt

# Two-stage
python scripts/evaluate_student.py \
  --checkpoint outputs/checkpoints_two_stage/student_final.pt \
  --two-stage
```

### Stage 5: Compare Teacher vs Student

See if distillation actually improved the student:

```bash
python scripts/evaluate_both.py \
  --teacher-checkpoint weights/teacher/teacher_finetuned_best.pth \
  --student-checkpoint outputs/checkpoints_two_stage/student_final.pt
# Output: Side-by-side comparison showing improvement
```

---

## Quick Start (Simple)

For a quick start without fine-tuning:

```bash
# 1. Prepare Data
mkdir -p dataset/{train,val}/{real,fake}
# Copy your images into these directories

# 2. Train Student with Pretrained Teacher
python scripts/train_student_two_stage.py \
  --dataset-root dataset \
  --teacher-weights wildrf \
  --device cuda

# 3. Evaluate
python scripts/evaluate_student.py \
  --checkpoint outputs/checkpoints_two_stage/student_final.pt \
  --two-stage
```

---

## Inference

```python
from inference.pipeline import InferencePipeline

# Load model
pipeline = InferencePipeline(student, device="cuda", threshold=0.5)

# Predict single image
result = pipeline.predict("image.jpg")
print(f"Fake probability: {result['fake_probability']:.4f}")
print(f"Is fake: {result['is_fake']}")

# Predict batch
results = pipeline.predict_batch(["img1.jpg", "img2.jpg"])
```

## Key Features

### Patch-Based Analysis
- Images processed as overlapping patches (224×224)
- Patch grid enforced via `PatchAdapter`
- Individual patch predictions aggregated with Top-K pooling

### Knowledge Distillation
- Large teacher (LaDeDa) guides lightweight student (TinyLaDeDa)
- Configurable temperature and loss weighting
- Supports both KL divergence and MSE distillation

### Quantization Auditing
- Dynamic range quantization (preferred)
- Full INT8 quantization (optional)
- Float vs quantized model comparison
- Tolerance-based validation

### Distribution Shift Robustness
- JPEG compression robustness testing
- Configurable quality levels
- Automatic threshold analysis across qualities

## Configuration Guide

### base.yaml
Defines frozen model architecture:
```yaml
model:
  teacher:
    pretrained: true
    freeze_backbone: false
  student:
    depth_multiplier: 0.5
    width_multiplier: 0.75

patches:
  patch_size: 224
  stride: 16
  enable_padding: true
```

### dataset.yaml
Dataset paths and preprocessing:
```yaml
dataset:
  root: "../dataset"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  resize_size: 224
```

### train.yaml
Training hyperparameters:
```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  scheduler: "cosine"

distillation:
  temperature: 4.0
  alpha: 0.5  # weight of KD loss
```

### quant.yaml
Quantization and threshold settings:
```yaml
quantization:
  strategy: "dynamic_range"
  bits: 8
  calibration_samples: 100

threshold:
  search_method: "grid"
  search_range: [0.3, 0.5, 0.7]
  search_metric: "f1"
```

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve
- **Accuracy@τ**: Accuracy at specific threshold τ

## Quantization Workflow

1. **Calibration**: Collect activation statistics (100 samples)
2. **Quantization**: Convert weights to INT8 (dynamic range)
3. **Validation**: Test quantized model on validation set
4. **Audit**: Compare float vs quantized predictions
5. **Threshold Tuning**: Find optimal threshold for quantized model

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

Key tests:
- `test_patch_alignment.py`: Verify patch grid correctness
- `test_topk_pooling.py`: Test Top-K aggregation
- `test_quant_consistency.py`: Check quantization fidelity
- `test_inference_contract.py`: Validate inference pipeline

## Deployment

Export student model for edge deployment:
```python
from scripts.export_tflite import export_to_tflite

export_to_tflite(student, "outputs/student.tflite")
```

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- scikit-learn >= 1.3.0
- Pillow >= 9.0.0
- OpenCV >= 4.8.0

## References

- LaDeDa: Teacher model architecture
- Knowledge Distillation: Hinton et al. (2015)
- Patch-based Detection: Local artifact analysis
- Quantization: Dynamic range quantization for edge deployment

## License

Team Converge Research Project

## Contact

For questions or contributions, please reach out to the Team Converge research group.
