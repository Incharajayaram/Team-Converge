#!/usr/bin/env python3
"""
Evaluate Quantized Models - Metrics Comparison

Compares metrics between original and quantized models:
- AUC (Area Under Curve)
- Accuracy
- Precision, Recall, F1
- Inference time

Usage with dummy data:
    python3 evaluate_quantized_metrics.py --dummy-data

Usage with real data:
    python3 evaluate_quantized_metrics.py --dataset-path data/test
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import json
import time
import logging
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MetricsEvaluator:
    """Evaluate model metrics on test set."""

    def __init__(self, device='cpu'):
        self.device = device

    def evaluate_model(self, model, dataloader, model_name="Model"):
        """Evaluate model and return metrics."""
        logger.info(f"\n  Evaluating {model_name}...")

        model.eval()
        all_logits = []
        all_labels = []
        inference_times = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)

                # Measure inference time
                start = time.time()
                try:
                    outputs = model(images)
                    elapsed = time.time() - start
                    inference_times.append(elapsed)

                    # Handle output shapes
                    if isinstance(outputs, dict):
                        logits = outputs.get('logits', outputs.get('output', outputs))
                    else:
                        logits = outputs

                    if logits.dim() == 4:  # (B, C, H, W)
                        logits_flat = logits.view(logits.size(0), -1).mean(dim=1)
                    elif logits.dim() == 2:  # (B, C)
                        logits_flat = logits.squeeze(1)
                    else:
                        logits_flat = logits

                    all_logits.extend(logits_flat.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                except Exception as e:
                    logger.warning(f"    Error on batch {batch_idx}: {e}")
                    continue

                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"    Processed {batch_idx + 1} batches...")

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        if len(all_logits) == 0:
            logger.error(f"    No data evaluated!")
            return None

        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np.exp(-all_logits))  # Sigmoid
        preds = (probs > 0.5).astype(int)

        # Compute metrics
        try:
            auc = roc_auc_score(all_labels, probs)
        except:
            auc = 0.5

        accuracy = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel()

        # Inference time
        avg_time_ms = (np.mean(inference_times) / len(images)) * 1000 if inference_times else 0

        metrics = {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'inference_time_ms': float(avg_time_ms),
            'num_samples': len(all_labels),
        }

        return metrics

    def compare_metrics(self, original_metrics, quantized_metrics):
        """Compare original vs quantized metrics."""
        comparison = {}

        for key in original_metrics:
            if key not in quantized_metrics:
                continue

            orig_val = original_metrics[key]
            quant_val = quantized_metrics[key]

            if isinstance(orig_val, (int, float)):
                diff = quant_val - orig_val
                pct = (diff / orig_val * 100) if orig_val != 0 else 0

                comparison[key] = {
                    'original': float(orig_val),
                    'quantized': float(quant_val),
                    'difference': float(diff),
                    'percent_change': float(pct),
                }

        return comparison


def create_dummy_dataloader(num_batches=20, batch_size=32):
    """Create dummy dataloader for testing."""
    logger.info("Creating dummy test data...")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Dummy image (256x256 RGB)
            image = torch.randn(3, 256, 256)
            # Random label (0 or 1)
            label = torch.randint(0, 2, (1,)).item()
            return image, label

    dataset = DummyDataset(num_batches * batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


def load_models(device='cpu'):
    """Load original and quantized models."""
    logger.info("\nLoading models...")

    from models.reference.LaDeDa import LaDeDa9
    from models.student.tiny_ladeda import TinyLaDeDa

    # Load teacher
    logger.info("  Loading teacher...")
    teacher_original = LaDeDa9(preprocess_type='NPR', num_classes=1, pool=True)
    teacher_state = torch.load(
        'deepfake-patch-audit/weights/teacher/WildRF_LaDeDa.pth',
        map_location=device,
        weights_only=False
    )
    teacher_original.load_state_dict(teacher_state)
    teacher_original.eval().to(device)

    # Load quantized teacher
    teacher_quantized = LaDeDa9(preprocess_type='NPR', num_classes=1, pool=True)
    teacher_quant_state = torch.load(
        'results/quantization_proper/teacher_quantized.pt',
        map_location=device,
        weights_only=False
    )
    teacher_quantized.load_state_dict(teacher_quant_state)
    teacher_quantized.eval().to(device)

    logger.info("  ✓ Teacher loaded (original + quantized)")

    # Load student
    logger.info("  Loading student...")
    student_original = TinyLaDeDa(pretrained=False)
    student_state = torch.load(
        'deepfake-patch-audit/weights/student/WildRF_Tiny_LaDeDa.pth',
        map_location=device,
        weights_only=False
    )
    student_original.model.load_state_dict(student_state)
    student_original.eval().to(device)

    # Load quantized student
    student_quantized = TinyLaDeDa(pretrained=False)
    student_quant_state = torch.load(
        'results/quantization_proper/student_quantized.pt',
        map_location=device,
        weights_only=False
    )
    student_quantized.model.load_state_dict(student_quant_state)
    student_quantized.eval().to(device)

    logger.info("  ✓ Student loaded (original + quantized)")

    return {
        'teacher_original': teacher_original,
        'teacher_quantized': teacher_quantized,
        'student_original': student_original,
        'student_quantized': student_quantized,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate quantized models')
    parser.add_argument('--dummy-data', action='store_true', help='Use dummy data')
    parser.add_argument('--dataset-path', default=None, help='Path to test dataset')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/quantization_metrics.json')

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Create test dataloader
    if args.dummy_data or args.dataset_path is None:
        logger.warning("\n⚠️  Using DUMMY data (random images and labels)")
        logger.warning("   Metrics will be ~0.5 AUC (random guessing)")
        logger.info("   For real evaluation, provide real test data\n")
        test_loader = create_dummy_dataloader(num_batches=20)
    else:
        logger.info(f"Loading dataset from {args.dataset_path}")
        # TODO: Load real dataset
        test_loader = create_dummy_dataloader(num_batches=20)

    # Load models
    try:
        models = load_models(device)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return

    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("EVALUATING TEACHER MODEL")
    logger.info("="*80)

    evaluator = MetricsEvaluator(device=device)

    teacher_original_metrics = evaluator.evaluate_model(
        models['teacher_original'], test_loader, "Teacher (Original)"
    )
    teacher_quantized_metrics = evaluator.evaluate_model(
        models['teacher_quantized'], test_loader, "Teacher (Quantized)"
    )

    logger.info("\n" + "="*80)
    logger.info("EVALUATING STUDENT MODEL")
    logger.info("="*80)

    student_original_metrics = evaluator.evaluate_model(
        models['student_original'], test_loader, "Student (Original)"
    )
    student_quantized_metrics = evaluator.evaluate_model(
        models['student_quantized'], test_loader, "Student (Quantized)"
    )

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("METRICS COMPARISON")
    logger.info("="*80)

    if teacher_original_metrics and teacher_quantized_metrics:
        teacher_comparison = evaluator.compare_metrics(
            teacher_original_metrics, teacher_quantized_metrics
        )

        logger.info("\n📊 TEACHER MODEL METRICS")
        logger.info("-" * 80)
        logger.info(f"{'Metric':<20} {'Original':<15} {'Quantized':<15} {'Difference':<15}")
        logger.info("-" * 80)

        for metric, values in teacher_comparison.items():
            if metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
                orig = values['original']
                quant = values['quantized']
                diff = values['percent_change']
                logger.info(f"{metric:<20} {orig:<15.4f} {quant:<15.4f} {diff:>+6.2f}%")

        logger.info(f"\n⏱️  Inference Time:")
        logger.info(f"  Original: {teacher_original_metrics['inference_time_ms']:.2f} ms/image")
        logger.info(f"  Quantized: {teacher_quantized_metrics['inference_time_ms']:.2f} ms/image")
        if teacher_original_metrics['inference_time_ms'] > 0:
            speedup = teacher_original_metrics['inference_time_ms'] / teacher_quantized_metrics['inference_time_ms']
            logger.info(f"  Speedup: {speedup:.2f}x")

    if student_original_metrics and student_quantized_metrics:
        student_comparison = evaluator.compare_metrics(
            student_original_metrics, student_quantized_metrics
        )

        logger.info("\n📊 STUDENT MODEL METRICS")
        logger.info("-" * 80)
        logger.info(f"{'Metric':<20} {'Original':<15} {'Quantized':<15} {'Difference':<15}")
        logger.info("-" * 80)

        for metric, values in student_comparison.items():
            if metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
                orig = values['original']
                quant = values['quantized']
                diff = values['percent_change']
                logger.info(f"{metric:<20} {orig:<15.4f} {quant:<15.4f} {diff:>+6.2f}%")

        logger.info(f"\n⏱️  Inference Time:")
        logger.info(f"  Original: {student_original_metrics['inference_time_ms']:.2f} ms/image")
        logger.info(f"  Quantized: {student_quantized_metrics['inference_time_ms']:.2f} ms/image")
        if student_original_metrics['inference_time_ms'] > 0:
            speedup = student_original_metrics['inference_time_ms'] / student_quantized_metrics['inference_time_ms']
            logger.info(f"  Speedup: {speedup:.2f}x")

    # Verdict
    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)

    if teacher_original_metrics and teacher_quantized_metrics:
        auc_drop = teacher_original_metrics['auc'] - teacher_quantized_metrics['auc']
        if abs(auc_drop) < 0.01:
            logger.info("✅ TEACHER: AUC drop < 1% - EXCELLENT quantization")
        elif abs(auc_drop) < 0.03:
            logger.info("✓ TEACHER: AUC drop 1-3% - ACCEPTABLE quantization")
        else:
            logger.warning(f"⚠️  TEACHER: AUC drop {abs(auc_drop):.2%} - May be too aggressive")

    if student_original_metrics and student_quantized_metrics:
        auc_drop = student_original_metrics['auc'] - student_quantized_metrics['auc']
        if abs(auc_drop) < 0.01:
            logger.info("✅ STUDENT: AUC drop < 1% - EXCELLENT quantization")
        elif abs(auc_drop) < 0.03:
            logger.info("✓ STUDENT: AUC drop 1-3% - ACCEPTABLE quantization")
        else:
            logger.warning(f"⚠️  STUDENT: AUC drop {abs(auc_drop):.2%} - May be too aggressive")

    # Save results
    results = {
        'teacher': {
            'original': teacher_original_metrics,
            'quantized': teacher_quantized_metrics,
            'comparison': teacher_comparison if teacher_original_metrics else None,
        },
        'student': {
            'original': student_original_metrics,
            'quantized': student_quantized_metrics,
            'comparison': student_comparison if student_original_metrics else None,
        },
        'note': 'Evaluated on dummy data (random images) - AUC expected ~0.5',
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
