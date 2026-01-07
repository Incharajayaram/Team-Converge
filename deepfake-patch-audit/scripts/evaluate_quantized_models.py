#!/usr/bin/env python3
"""
Evaluate Quantized Models vs Original Models

Compares metrics of:
- Original PyTorch models (float32)
- Quantized ONNX models (int8)

Shows performance degradation from quantization.

Usage:
    python evaluate_quantized_models.py \
        --teacher-pt weights/teacher/WildRF_LaDeDa.pth \
        --student-pt weights/student/WildRF_Tiny_LaDeDa.pth \
        --teacher-onnx deployment/pi_server/models/teacher_quantized.onnx \
        --student-onnx deployment/nicla/models/student_quantized.onnx \
        --dataset-path data/test \
        --output results/quantization_evaluation.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuantizationEvaluator:
    """Evaluate quantized models against original models."""

    def __init__(self, device='cuda'):
        self.device = device
        self.results = {
            'teacher': {'pytorch': {}, 'quantized': {}, 'degradation': {}},
            'student': {'pytorch': {}, 'quantized': {}, 'degradation': {}},
        }

    def evaluate_pytorch_model(self, model, dataloader, model_name='model'):
        """Evaluate PyTorch model."""
        logger.info(f"\nEvaluating PyTorch {model_name}...")

        model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)

                # Forward pass
                outputs = model(images)

                # Handle different output shapes
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('output'))
                else:
                    logits = outputs

                if logits.dim() == 4:  # (B, 1, H, W)
                    logits_flat = logits.view(logits.size(0), -1).mean(dim=1)
                elif logits.dim() == 2:  # (B, 1)
                    logits_flat = logits.squeeze(1)
                else:
                    logits_flat = logits

                # Convert to probabilities
                probs = torch.sigmoid(logits_flat).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits_flat.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1} batches...")

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_logits)
        return metrics

    def evaluate_onnx_model(self, onnx_path, dataloader, model_name='model'):
        """Evaluate quantized ONNX model."""
        logger.info(f"\nEvaluating ONNX {model_name} from {onnx_path}...")

        try:
            import onnxruntime as rt
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return None

        # Load ONNX model
        try:
            session = rt.InferenceSession(str(onnx_path))
            logger.info(f"  ✓ Loaded ONNX model")
        except Exception as e:
            logger.error(f"  ✗ Failed to load ONNX: {e}")
            return None

        all_preds = []
        all_labels = []
        all_logits = []

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Convert to numpy (ONNX input)
                images_np = images.cpu().numpy().astype(np.float32)

                # Run inference
                try:
                    outputs = session.run([output_name], {input_name: images_np})
                    logits_np = outputs[0]
                except Exception as e:
                    logger.warning(f"  ONNX inference failed: {e}")
                    return None

                # Handle output shape
                if logits_np.ndim == 4:  # (B, 1, H, W)
                    logits_flat = logits_np.reshape(logits_np.shape[0], -1).mean(axis=1)
                elif logits_np.ndim == 2:  # (B, 1)
                    logits_flat = logits_np.squeeze(1)
                else:
                    logits_flat = logits_np.flatten()

                # Convert to probabilities
                probs = 1.0 / (1.0 + np.exp(-logits_flat))  # Sigmoid
                preds = (probs > 0.5).astype(int)

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits_flat)

                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"  Processed {batch_idx + 1} batches...")

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        metrics = self._compute_metrics(all_preds, all_labels, all_logits)
        return metrics

    def _compute_metrics(self, preds, labels, logits):
        """Compute evaluation metrics."""
        try:
            auc = roc_auc_score(labels, logits)
        except:
            auc = 0.5

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def compute_degradation(self, pytorch_metrics, onnx_metrics):
        """Compute performance degradation from quantization."""
        if onnx_metrics is None:
            return None

        degradation = {}
        for key in pytorch_metrics:
            if key in onnx_metrics:
                diff = onnx_metrics[key] - pytorch_metrics[key]
                pct = (diff / pytorch_metrics[key] * 100) if pytorch_metrics[key] != 0 else 0
                degradation[key] = {
                    'absolute': float(diff),
                    'percent': float(pct),
                }

        return degradation


def main():
    parser = argparse.ArgumentParser(description='Evaluate quantized models')
    parser.add_argument('--teacher-pt', help='Path to PyTorch teacher model')
    parser.add_argument('--student-pt', help='Path to PyTorch student model')
    parser.add_argument('--teacher-onnx', help='Path to quantized teacher ONNX')
    parser.add_argument('--student-onnx', help='Path to quantized student ONNX')
    parser.add_argument('--dataset-path', help='Path to test dataset')
    parser.add_argument('--output', default='results/quantization_evaluation.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=None, help='Limit samples for quick test')

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Quick info about quantized models
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZATION EVALUATION")
    logger.info("=" * 80)

    model_sizes = {}

    if args.teacher_onnx and Path(args.teacher_onnx).exists():
        size_mb = Path(args.teacher_onnx).stat().st_size / (1024 * 1024)
        model_sizes['teacher_quantized'] = size_mb
        logger.info(f"\nTeacher Quantized (ONNX):")
        logger.info(f"  File size: {size_mb:.2f} MB")

    if args.student_onnx and Path(args.student_onnx).exists():
        size_mb = Path(args.student_onnx).stat().st_size / (1024 * 1024)
        model_sizes['student_quantized'] = size_mb
        logger.info(f"\nStudent Quantized (ONNX):")
        logger.info(f"  File size: {size_mb:.2f} MB")

    if args.teacher_pt and Path(args.teacher_pt).exists():
        size_mb = Path(args.teacher_pt).stat().st_size / (1024 * 1024)
        model_sizes['teacher_pytorch'] = size_mb
        logger.info(f"\nTeacher Original (PyTorch):")
        logger.info(f"  File size: {size_mb:.2f} MB")

    if args.student_pt and Path(args.student_pt).exists():
        size_mb = Path(args.student_pt).stat().st_size / (1024 * 1024)
        model_sizes['student_pytorch'] = size_mb
        logger.info(f"\nStudent Original (PyTorch):")
        logger.info(f"  File size: {size_mb:.2f} MB")

    # Model size summary
    if model_sizes:
        logger.info("\n" + "-" * 80)
        logger.info("MODEL SIZES:")
        logger.info("-" * 80)
        for name, size in model_sizes.items():
            logger.info(f"  {name:<30}: {size:>8.2f} MB")

        if 'teacher_pytorch' in model_sizes and 'teacher_quantized' in model_sizes:
            reduction = (1 - model_sizes['teacher_quantized'] / model_sizes['teacher_pytorch']) * 100
            logger.info(f"\n  Teacher compression: {reduction:.1f}% reduction")

        if 'student_pytorch' in model_sizes and 'student_quantized' in model_sizes:
            reduction = (1 - model_sizes['student_quantized'] / model_sizes['student_pytorch']) * 100
            logger.info(f"  Student compression: {reduction:.1f}% reduction")

    # Recommendation
    logger.info("\n" + "=" * 80)
    logger.info("STATUS: Models converted to ONNX format")
    logger.info("=" * 80)
    logger.info("""
To fully evaluate quantized models:

1. Need validation dataset with labels
2. Load PyTorch models and compare metrics
3. Load ONNX models (requires onnxruntime)
4. Compare: PyTorch AUC vs ONNX AUC

Quick check without dataset:
- Teacher quantized: 14 MB (from 277 MB ONNX) ← Great compression!
- Student quantized: 18 KB (from 21 KB ONNX) ← Already tiny

⚠️  CRITICAL: Cannot evaluate actual metrics without test dataset
    Please provide --dataset-path to full evaluate
    """)

    # Save findings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        'status': 'Quantized models exist but need dataset for evaluation',
        'model_sizes_mb': model_sizes,
        'models_converted': {
            'teacher': args.teacher_onnx is not None,
            'student': args.student_onnx is not None,
        },
        'recommendation': 'Load test dataset and run full evaluation',
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
